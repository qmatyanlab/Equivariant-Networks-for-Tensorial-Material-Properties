"""Metrics, bootstrap confidence intervals, paired tests, derived elastic moduli and task similarity.

Reads the .npz files written by eval_dump.py; light numpy work (one core, seconds).

Usage:
  python analyze.py <npz_dir> <out_dir> [--n-boot 1000] [--seed 0]

Run names follow R2026_<Arch>_-_<Target>_-_<Variant>, with Target in {Elastic, Dielectric, Piezo} and Variant in
{Default, Pretrained (band gap), Pretrained_-_Formation_Energy, From_Elastic}.

Writes to out_dir:
  metrics.csv     one row per run: MAE [95% CI], RMSE, R2, per-irrep MAE; piezo: allowed-component MAE, zero baselines;
                  relative errors: mean |DFT component| and MAE relative to it (nMAE); per-material relative error
                  ||T_pred - T_DFT|| / ||T_DFT|| in the Frobenius norm of the full Cartesian tensor (median, IQR, pooled
                  ratio; piezo median only over tensors with ||T_DFT|| >= --piezo-min-norm)
  relerr.csv      per-material relative errors of every run (for the SI histogram)
  paired.csv      paired bootstrap comparisons (pretraining vs scratch per architecture; architectures vs each other),
                  with Holm-adjusted p-values over all comparisons
  moduli.csv      elastic runs: MAE [95% CI] of K_H, G_H, E derived from the predicted tensors, and their relative
                  errors: MAPE and median APE [95% CI], fraction of materials within 10% and 20%
  similarity.json Pearson correlation of per-material errors between tasks on shared test materials; linear CKA
                  between backbone embeddings on the common probe set
  tables.tex      LaTeX rows for the direct, transfer and moduli tables
"""
import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

TARGET_OF = {"Elastic": "elastic", "Dielectric": "dielectric", "Piezo": "piezoelectric"}
VARIANT_LABEL = {"Default": "none", "Pretrained": "band gap", "Pretrained_-_Formation_Energy": "formation energy",
                 "From_Elastic": "elasticity"}
UNIT = {"elastic": "GPa", "dielectric": "", "piezoelectric": "C/m^2"}
# irreps blocks of the de-normalized coefficient vectors (order as in the model output)
IRREP_BLOCKS = {
    "elastic": [("S 0e", 0, 1), ("S 2e", 1, 6), ("S 4e", 6, 15), ("A 0e", 15, 16), ("A 2e", 16, 21)],
    "dielectric": [("0e", 0, 1), ("2e", 1, 6)],
    "piezoelectric": [("S 1o", 0, 3), ("S 3o", 3, 10), ("A 1o", 10, 13), ("A 2o", 13, 18)],
}


def parse_name(name):
    m = re.match(r"R2026_(SEGNN|SEConv|SETransformer)_-_(Elastic|Dielectric|Piezo)_-_(.+)$", name)
    if not m:
        return None
    arch, tgt, var = m.groups()
    return arch, TARGET_OF[tgt], var


def unique_components(target, x):
    """(N, ...) Cartesian/Voigt tensors -> (N, n_independent)."""
    if target == "elastic":
        iu = np.triu_indices(6)
        return x[:, iu[0], iu[1]]
    if target == "dielectric":
        iu = np.triu_indices(3)
        return x[:, iu[0], iu[1]]
    return x.reshape(len(x), 18)


def per_material_mae(target, pred, true, mask=None):
    e = np.abs(unique_components(target, pred) - unique_components(target, true))
    if mask is None:
        return e.mean(1)
    m = unique_components(target, mask.astype(float))
    return (e * m).sum(1) / np.maximum(m.sum(1), 1)


# Voigt -> Cartesian weights: a Voigt index 4-6 stands for two Cartesian index pairs (23 and 32, ...), and for the
# stress constants and the stiffness the Voigt entries equal the tensor components, so each squared entry counts
# 1 or 2 times per Voigt index.
_VW = np.array([1.0, 1, 1, 2, 2, 2])
FROB_W = {"dielectric": np.ones((3, 3)), "piezoelectric": np.tile(_VW, (3, 1)), "elastic": np.outer(_VW, _VW)}


def frobenius(target, x):
    """(N, ...) tensors in the stored form (3x3, 3x6 Voigt, 6x6 Voigt) -> (N,) Frobenius norm of the full Cartesian
    tensor (the norm of the frame residual in frame_align.py)."""
    return np.sqrt((FROB_W[target] * np.asarray(x, float) ** 2).reshape(len(x), -1).sum(1))


def ratio_ci(num, den, idx):
    """sum(num)/sum(den) with a bootstrap CI over materials (pooled relative error, nMAE)."""
    boots = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def stat_ci(values, idx, stat):
    """stat (NaN-aware, e.g. np.nanmedian) of per-material values with a bootstrap CI; NaN entries are left out."""
    boots = np.array([stat(values[i]) for i in idx])
    return float(stat(values)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def bootstrap_ci(values, idx):
    """values: (N,) per-material errors; idx: (B, N) resample indices. Returns mean, lo, hi."""
    boots = values[idx].mean(1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_test(a, b, n_boot, seed, chunk=1000):
    """Mean of a - b with 95% CI and a two-sided paired-bootstrap p-value (a, b aligned per material).

    The resamples are drawn in chunks (memory) from a generator seeded per target, so every pair of the same target sees
    the same resamples. p = 2 min(#{d* <= 0} + 1, #{d* >= 0} + 1) / (B + 1), which is never zero: with B resamples the
    smallest p is about 2/B, so B must be large enough for the Holm adjustment over the whole family (m p_min << 0.05).
    """
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, len(d), size=(min(chunk, n_boot - k), len(d)))].mean(1)
                            for k in range(0, n_boot, chunk)])
    p = 2 * min(np.sum(boots <= 0) + 1, np.sum(boots >= 0) + 1) / (n_boot + 1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(min(p, 1.0))


def holm(p):
    """Holm-Bonferroni adjusted p-values (step-down, monotone), same order as the input."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def positive_definite(C):
    """True where the (N, 6, 6) Voigt matrix is positive definite. The Mandel matrix W C W (W = diag(1,1,1,√2,√2,√2))
    is congruent to it, so the test is the same; its eigenvalues are those of the stiffness as a 4th-order tensor."""
    w = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    C = np.asarray(C, float)
    M = w[:, None] * ((C + np.transpose(C, (0, 2, 1))) / 2) * w[None, :]
    return np.linalg.eigvalsh(M)[:, 0] > 0


def moduli(C):
    """Hill-averaged K, G and E (GPa) from (N, 6, 6) Voigt elasticity matrices; NaN where C is singular."""
    C = np.asarray(C, float)
    KV = (C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2] + 2 * (C[:, 0, 1] + C[:, 1, 2] + C[:, 2, 0])) / 9
    GV = (C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2] - (C[:, 0, 1] + C[:, 1, 2] + C[:, 2, 0])
          + 3 * (C[:, 3, 3] + C[:, 4, 4] + C[:, 5, 5])) / 15
    S = np.full_like(C, np.nan)
    for i, c in enumerate(C):
        try:
            S[i] = np.linalg.inv(c)
        except np.linalg.LinAlgError:
            pass
    KR = 1 / (S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2] + 2 * (S[:, 0, 1] + S[:, 1, 2] + S[:, 2, 0]))
    GR = 15 / (4 * (S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2]) - 4 * (S[:, 0, 1] + S[:, 1, 2] + S[:, 2, 0])
               + 3 * (S[:, 3, 3] + S[:, 4, 4] + S[:, 5, 5]))
    K, G = (KV + KR) / 2, (GV + GR) / 2
    return {"K_H": K, "G_H": G, "E": 9 * K * G / (3 * K + G)}


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else float("nan")


def fmt(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-test", type=int, default=20000, help="paired-bootstrap resamples for the p-values")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piezo-min-norm", type=float, default=0.1,
                    help="C/m^2; per-material relative errors of piezoelectric tensors smaller than this are not used "
                         "for the median (near-zero tensors make the ratio meaningless); the pooled ratio uses all")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    runs = {}
    for f in sorted(Path(a.npz_dir).glob("*.npz")):
        info = parse_name(f.stem)
        if info:
            d = dict(np.load(f, allow_pickle=False))
            if info[1] == "dielectric":
                # reference = symmetric part of the stored DFT tensor (a few stored tensors are slightly asymmetric;
                # the 0e + 2e output represents only the symmetric part)
                d["true"] = (d["true"] + np.swapaxes(d["true"], -1, -2)) / 2
            runs[f.stem] = (info, d)
    if not runs:
        raise SystemExit("no R2026_* npz files found")

    # one resample index set per target, shared by all runs of that target (same test materials, same order)
    boot_idx, ids_of = {}, {}
    for name, ((arch, tgt, var), d) in runs.items():
        ids = list(d["ids"])
        if tgt in ids_of:
            assert ids == ids_of[tgt], f"{name}: test-set order differs from other {tgt} runs"
        else:
            ids_of[tgt] = ids
            boot_idx[tgt] = rng.integers(0, len(ids), size=(a.n_boot, len(ids)))

    # metrics
    err, rows, relerr = {}, [], {}
    for name, ((arch, tgt, var), d) in runs.items():
        idx = boot_idx[tgt]
        e = per_material_mae(tgt, d["pred"], d["true"])
        err[name] = e
        mae, lo, hi = bootstrap_ci(e, idx)
        P, T = unique_components(tgt, d["pred"]), unique_components(tgt, d["true"])
        row = {"run": name, "arch": arch, "target": tgt, "pretraining": VARIANT_LABEL.get(var, var), "n": len(e),
               "unit": UNIT[tgt], "MAE": mae, "MAE_lo": lo, "MAE_hi": hi,
               "RMSE": float(np.sqrt(np.mean((P - T) ** 2))),
               "R2": float(1 - np.sum((P - T) ** 2) / np.sum((T - T.mean()) ** 2))}
        # relative errors. (1) nMAE: MAE over the mean |DFT component| (same averaging), as a fraction.
        abs_true = np.abs(T).mean(1)
        row["mean_abs_true"] = float(abs_true.mean())
        row["nMAE"], row["nMAE_lo"], row["nMAE_hi"] = ratio_ci(e, abs_true, idx)
        # (2) per-material relative error in the Frobenius norm of the Cartesian tensor
        dn, tn = frobenius(tgt, d["pred"] - d["true"]), frobenius(tgt, d["true"])
        rel = np.where(tn > 0, dn / np.where(tn > 0, tn, 1), np.nan)
        if tgt == "piezoelectric":
            rel = np.where(tn >= a.piezo_min_norm, rel, np.nan)
        relerr[name] = (d["ids"], tn, rel)
        row["rel_n"] = int(np.isfinite(rel).sum())
        row["rel_median"], row["rel_median_lo"], row["rel_median_hi"] = stat_ci(rel, idx, np.nanmedian)
        row["rel_q25"], row["rel_q75"] = (float(q) for q in np.nanpercentile(rel, [25, 75]))
        row["rel_pooled"], row["rel_pooled_lo"], row["rel_pooled_hi"] = ratio_ci(dn, tn, idx)
        for lab, i0, i1 in IRREP_BLOCKS[tgt]:
            eb = np.abs(d["pred_coef"][:, i0:i1] - d["true_coef"][:, i0:i1]).mean(1)
            row[f"MAE_coef[{lab}]"], row[f"MAE_coef[{lab}]_lo"], row[f"MAE_coef[{lab}]_hi"] = bootstrap_ci(eb, idx)
        if tgt == "piezoelectric":
            ea = per_material_mae(tgt, d["pred"], d["true"], d["allowed"])
            row["MAE_allowed"], row["MAE_allowed_lo"], row["MAE_allowed_hi"] = bootstrap_ci(ea, idx)
            zero = np.zeros_like(d["true"])
            row["MAE_zero_baseline"] = float(per_material_mae(tgt, zero, d["true"]).mean())
            row["MAE_allowed_zero_baseline"] = float(per_material_mae(tgt, zero, d["true"], d["allowed"]).mean())
            # largest predicted magnitude among the symmetry-forbidden components (equivariance check on real data)
            forb = ~unique_components(tgt, d["allowed"]).astype(bool)
            row["max_abs_forbidden_pred"] = float(np.abs(P[forb]).max()) if forb.any() else 0.0
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union of columns, first-seen order
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    with open(out / "relerr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "id", "frobenius_true", "rel_err"])
        for name, (ids, tn, rel) in relerr.items():
            for i, t_, r_ in zip(ids, tn, rel):
                w.writerow([name, i, float(t_), float(r_)])

    # paired comparisons
    by = {(i[0], i[1], i[2]): n for n, (i, _) in runs.items()}
    pairs = []
    for (arch, tgt, var), n in by.items():
        base = by.get((arch, tgt, "Default"))
        if var != "Default" and base:
            pairs.append((n, base, f"{VARIANT_LABEL.get(var, var)} vs none", tgt))
    for tgt in set(t for _, t, _ in by):
        for a1, a2 in itertools.combinations(["SEGNN", "SEConv", "SETransformer"], 2):
            n1, n2 = by.get((a1, tgt, "Default")), by.get((a2, tgt, "Default"))
            if n1 and n2:
                pairs.append((n1, n2, f"{a1} vs {a2} (scratch)", tgt))
    with open(out / "paired.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_a", "run_b", "comparison", "target", "mean_diff_a_minus_b", "ci_lo", "ci_hi", "p_value",
                    "p_holm"])
        # Holm correction over the whole family of comparisons (transfer vs scratch and architecture pairs, all targets).
        # p-values from --n-boot-test resamples (smallest p ~ 2/B; with B = 20,000 and 33 comparisons, 33 x 1e-4 = 0.0033).
        res = [paired_test(err[n1], err[n2], a.n_boot_test, [a.seed, 1, sorted(ids_of).index(t)]) for n1, n2, _, t in pairs]
        adj = holm([r[3] for r in res]) if res else []
        for (n1, n2, lab, tgt), r, ph in zip(pairs, res, adj):
            w.writerow([n1, n2, lab, tgt, *r, float(ph)])

    # derived elastic moduli
    mod_rows, mod_err = [], {}
    for name, ((arch, tgt, var), d) in runs.items():
        if tgt != "elastic":
            continue
        mp, mt = moduli(d["pred"]), moduli(d["true"])
        # The Hill moduli need the compliance (Reuss terms); for a predicted tensor that is not positive definite they
        # have no physical meaning and produce the largest outliers, so such materials are excluded from the main
        # moduli statistics (decision 2026-11-04); the *_all columns keep every material (SI).
        pd = positive_definite(d["pred"])
        r = {"run": name, "arch": arch, "pretraining": VARIANT_LABEL.get(var, var), "n_nonPD": int((~pd).sum())}
        idx = boot_idx["elastic"]
        for k in mp:
            e_all = np.abs(mp[k] - mt[k])
            e = np.where(pd, e_all, np.nan)
            ok = np.isfinite(e)
            boots = np.array([np.nanmean(e[i]) for i in idx])
            r.update({k: float(np.nanmean(e)), f"{k}_lo": float(np.percentile(boots, 2.5)),
                      f"{k}_hi": float(np.percentile(boots, 97.5)), f"{k}_n_invalid": int((~ok).sum())})
            r[f"{k}_all"], r[f"{k}_all_lo"], r[f"{k}_all_hi"] = stat_ci(e_all, idx, np.nanmean)
            # relative errors (the DFT moduli are positive: cleaned tensors are positive definite)
            ape = e / np.abs(mt[k])
            r[f"{k}_MAPE"], r[f"{k}_MAPE_lo"], r[f"{k}_MAPE_hi"] = stat_ci(ape, idx, np.nanmean)
            r[f"{k}_MdAPE"], r[f"{k}_MdAPE_lo"], r[f"{k}_MdAPE_hi"] = stat_ci(ape, idx, np.nanmedian)
            for pct in (10, 20):
                within = np.where(np.isfinite(ape), (ape <= pct / 100).astype(float), np.nan)
                r[f"{k}_within{pct}"], r[f"{k}_within{pct}_lo"], r[f"{k}_within{pct}_hi"] = stat_ci(within, idx,
                                                                                                    np.nanmean)
        mod_rows.append(r)
        mod_err[name] = {k: np.where(pd, np.abs(mp[k] - mt[k]), np.nan) for k in mp}
    if mod_rows:
        with open(out / "moduli.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mod_rows[0]))
            w.writeheader()
            w.writerows(mod_rows)
        # paired tests of the moduli errors: the elastic pairs of the tensor family, per modulus; Holm over this family
        # (pairs x 3 moduli), separate from the 33 tensor comparisons. Materials with an undefined modulus in either
        # run are dropped from that comparison.
        mpairs = [(n1, n2, lab, k) for n1, n2, lab, t in pairs if t == "elastic" for k in ("K_H", "G_H", "E")]
        mres = []
        for n1, n2, _, k in mpairs:
            a1, a2 = mod_err[n1][k], mod_err[n2][k]
            ok = np.isfinite(a1) & np.isfinite(a2)
            mres.append(paired_test(a1[ok], a2[ok], a.n_boot_test, [a.seed, 2, ("K_H", "G_H", "E").index(k)]))
        madj = holm([r[3] for r in mres]) if mres else []
        with open(out / "paired_moduli.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_a", "run_b", "comparison", "modulus", "mean_diff_a_minus_b", "ci_lo", "ci_hi", "p_value",
                        "p_holm"])
            for (n1, n2, lab, k), r, ph in zip(mpairs, mres, madj):
                w.writerow([n1, n2, lab, k, *r, float(ph)])

    # similarity: per-material error correlation on shared test materials, and CKA on the probe set
    sim = {"error_correlation": [], "cka": {}}
    names = list(runs)
    for n1, n2 in itertools.combinations(names, 2):
        (i1, d1), (i2, d2) = runs[n1], runs[n2]
        if i1[1] == i2[1]:
            continue
        common = sorted(set(d1["ids"]) & set(d2["ids"]))
        if len(common) < 20:
            continue
        p1 = {k: v for k, v in zip(d1["ids"], err[n1])}
        p2 = {k: v for k, v in zip(d2["ids"], err[n2])}
        x, y = np.array([p1[c] for c in common]), np.array([p2[c] for c in common])
        sim["error_correlation"].append({"a": n1, "b": n2, "n_shared": len(common),
                                         "pearson": float(np.corrcoef(x, y)[0, 1])})
    with_probe = [n for n in names if "probe_emb" in runs[n][1]]
    for n1, n2 in itertools.combinations(with_probe, 2):
        assert list(runs[n1][1]["probe_ids"]) == list(runs[n2][1]["probe_ids"])
        sim["cka"][f"{n1}|{n2}"] = linear_cka(runs[n1][1]["probe_emb"], runs[n2][1]["probe_emb"])
    (out / "similarity.json").write_text(json.dumps(sim, indent=1))

    # LaTeX rows
    lines = ["% direct prediction (training from scratch): MAE [95% CI]"]
    for tgt in ["elastic", "dielectric", "piezoelectric"]:
        cells = []
        for arch in ["SEGNN", "SEConv", "SETransformer"]:
            r = next((r for r in rows if r["target"] == tgt and r["arch"] == arch and r["pretraining"] == "none"), None)
            cells.append(fmt(r["MAE"], r["MAE_lo"], r["MAE_hi"], 4 if tgt == "piezoelectric" else 3) if r else "--")
        lines.append(f"{tgt} & " + " & ".join(cells) + r" \\")
    lines.append("% direct prediction: mean |DFT component|, nMAE (%) [95% CI], median relative error (%) [95% CI]")
    for tgt in ["elastic", "dielectric", "piezoelectric"]:
        cells = []
        for arch in ["SEGNN", "SEConv", "SETransformer"]:
            r = next((r for r in rows if r["target"] == tgt and r["arch"] == arch and r["pretraining"] == "none"), None)
            cells.append(f"{fmt(100 * r['nMAE'], 100 * r['nMAE_lo'], 100 * r['nMAE_hi'], 1)}; "
                         f"{fmt(100 * r['rel_median'], 100 * r['rel_median_lo'], 100 * r['rel_median_hi'], 1)}"
                         if r else "--")
        r0 = next((r for r in rows if r["target"] == tgt), None)
        lines.append(f"{tgt} ({r0['mean_abs_true']:.4g}) & " + " & ".join(cells) + r" \\" if r0 else f"% {tgt}: none")
    lines.append("% transfer: MAE [95% CI] by pretraining")
    for tgt in ["elastic", "dielectric", "piezoelectric"]:
        for pre in ["none", "band gap", "formation energy", "elasticity"]:
            cells = []
            for arch in ["SEGNN", "SEConv", "SETransformer"]:
                r = next((r for r in rows if r["target"] == tgt and r["arch"] == arch and r["pretraining"] == pre), None)
                cells.append(fmt(r["MAE"], r["MAE_lo"], r["MAE_hi"], 4 if tgt == "piezoelectric" else 3) if r else "--")
            if any(c != "--" for c in cells):
                lines.append(f"{tgt} & {pre} & " + " & ".join(cells) + r" \\")
    if mod_rows:
        lines.append("% moduli: K_H, G_H, E MAE [95% CI] (GPa)")
        for r in mod_rows:
            lines.append(f"{r['arch']} & {r['pretraining']} & " + " & ".join(
                fmt(r[k], r[f'{k}_lo'], r[f'{k}_hi'], 2) for k in ["K_H", "G_H", "E"]) + r" \\")
        lines.append("% moduli: K_H, G_H, E MAPE (%) [95% CI]; fraction within 10% / 20%")
        for r in mod_rows:
            lines.append(f"{r['arch']} & {r['pretraining']} & " + " & ".join(
                f"{fmt(100 * r[k + '_MAPE'], 100 * r[k + '_MAPE_lo'], 100 * r[k + '_MAPE_hi'], 1)}; "
                f"{100 * r[k + '_within10']:.0f}/{100 * r[k + '_within20']:.0f}" for k in ["K_H", "G_H", "E"]) + r" \\")
    (out / "tables.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(runs)} runs analyzed -> {out}")


if __name__ == "__main__":
    main()
