"""Write the LaTeX tables of the revision from the outputs of analyze.py (stdlib only; no hand transcription).

  python scripts/make_tables.py analysis/eval manuscript/tables --final elastic
  python scripts/make_tables.py analysis/eval manuscript/tables --final elastic dielectric piezoelectric

Only targets listed in --final are filled; cells of the other targets are written as \\todo{} (their models are being
retrained), so a table never mixes final and provisional numbers silently. Holm-adjusted p-values depend on the whole
family of 33 comparisons, so p-values and significance marks are only final when all three targets are final; until
then they are wrapped in \\prov{...} (defined in macros.tex: marked in the revision PDF).

Outputs (one file per table body, \\input by the manuscript and the SI):
  direct.tex, moduli.tex, transfer.tex                    main-text Tables 1, 3, 4
  si_irrep.tex, si_equiv_trained.tex, si_transfer.tex,     Supplementary Tables S4, S9 (trained column), S12, S13,
  si_archs.tex, si_allmetrics.tex, si_moduli_paired.tex,   S14, S15, S16
  si_moduli_all.tex
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

ARCHS = ["SEGNN", "SEConv", "SETransformer"]
TARGETS = ["elastic", "dielectric", "piezoelectric"]
TNAME = {"elastic": "Elasticity", "dielectric": "Dielectric", "piezoelectric": "Piezoelectric"}
UNIT = {"elastic": "GPa", "dielectric": "", "piezoelectric": "C\\,m$^{-2}$"}
DEC = {"elastic": 2, "dielectric": 2, "piezoelectric": 4}
BACKBONES = {"elastic": ["none", "band gap", "formation energy"],
             "dielectric": ["none", "band gap", "formation energy", "elasticity"],
             "piezoelectric": ["none", "band gap", "formation energy", "elasticity"]}
BBNAME = {"none": "none (scratch)", "band gap": "band gap", "formation energy": "formation energy",
          "elasticity": "elasticity"}
IRREPS = {"dielectric": [("0e", "\\irrep{0e}"), ("2e", "\\irrep{2e}")],
          "piezoelectric": [("S 1o", "$S$: \\irrep{1o}"), ("S 3o", "$S$: \\irrep{3o}"), ("A 1o", "$A$: \\irrep{1o}"),
                            ("A 2o", "$A$: \\irrep{2o}")],
          "elastic": [("S 0e", "$S$: \\irrep{0e}"), ("S 2e", "$S$: \\irrep{2e}"), ("S 4e", "$S$: \\irrep{4e}"),
                      ("A 0e", "$A$: \\irrep{0e}"), ("A 2e", "$A$: \\irrep{2e}")]}
TODO = "\\todo{}"


def f(x, d):
    return f"{float(x):.{d}f}"


def ci(r, k, d):
    return f"{f(r[k], d)} [{f(r[k + '_lo'], d)}, {f(r[k + '_hi'], d)}]"


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


N_BOOT_TEST = 20000  # analyze.py --n-boot-test


def pfmt(p, raw=None):
    """Holm-adjusted p. When the raw p is at the resolution floor of the paired bootstrap (2/(B+1): no resample crossed
    zero), the adjusted value is an upper bound and is printed with a "<=" sign."""
    p = float(p)
    le = raw is not None and float(raw) <= 2 / (N_BOOT_TEST + 1) * (1 + 1e-9)
    if p >= 0.995:
        return "1"
    v = f"{p:.2g}" if p < 0.1 else f"{p:.2f}"
    return ("$\\le$\\," if le else "") + v


def signed(x, d):
    x = float(x)
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.{d}f}"


class Data:
    def __init__(self, d, final):
        self.final = set(final)
        self.all_final = self.final >= set(TARGETS)
        self.m = {(r["arch"], r["target"], r["pretraining"]): r for r in csv.DictReader(open(d / "metrics.csv"))}
        self.mod = {(r["arch"], r["pretraining"]): r for r in csv.DictReader(open(d / "moduli.csv"))}
        self.pairs = list(csv.DictReader(open(d / "paired.csv")))
        pm = d / "paired_moduli.csv"
        self.mpairs = list(csv.DictReader(open(pm))) if pm.exists() else []
        self.equiv = defaultdict(lambda: defaultdict(float))  # (arch, target) -> op -> max error
        eq = d / "equivariance" / "equivariance.txt"
        if eq.exists():
            for line in eq.read_text().splitlines():
                m = re.match(r"R2026_(\w+?)_-_(Elastic|Dielectric|Piezo)_-_.* (rotation|inversion|rotoinversion) vs .*= ([0-9.e+-]+)", line)
                if m:
                    arch, t, op, v = m.groups()
                    t = {"Elastic": "elastic", "Dielectric": "dielectric", "Piezo": "piezoelectric"}[t]
                    self.equiv[(arch, t)][op] = max(self.equiv[(arch, t)][op], float(v))

    def ok(self, t):
        return t in self.final

    def prov(self, s):
        return s if self.all_final else f"\\prov{{{s}}}"

    def pair(self, t, comparison, arch=None):
        for p in self.pairs:
            if p["target"] == t and p["comparison"] == comparison and (arch is None or f"R2026_{arch}_-_" in p["run_a"]):
                return p
        return None


def direct(D):
    rows = [("MAE [95\\% CI]", lambda r, t: ci(r, "MAE", DEC[t])),
            ("RMSE", lambda r, t: f(r["RMSE"], DEC[t])),
            ("$R^2$", lambda r, t: f(r["R2"], 3)),
            ("nMAE (\\%)", lambda r, t: f"{pct(r['nMAE'])} [{pct(r['nMAE_lo'])}, {pct(r['nMAE_hi'])}]"),
            ("Median relative error (\\%)", lambda r, t: f"{pct(r['rel_median'])} [{pct(r['rel_median_lo'])}, "
                                                          f"{pct(r['rel_median_hi'])}]")]
    piezo_extra = [("MAE, allowed components", lambda r, t: ci(r, "MAE_allowed", DEC[t])),
                   ("Predict-zero baseline, all / allowed",
                    lambda r, t: f"{f(r['MAE_zero_baseline'], DEC[t])} / {f(r['MAE_allowed_zero_baseline'], DEC[t])}")]
    out = ["\\begin{tabular}{llccc}", "\\toprule", "Target & Metric & SEGNN & SEConv & SETransformer \\\\", "\\midrule"]
    for k, t in enumerate(TARGETS):
        label = TNAME[t] + (f" ({UNIT[t]})" if UNIT[t] else "")
        for j, (name, fn) in enumerate(rows + (piezo_extra if t == "piezoelectric" else [])):
            cells = [fn(D.m[(a, t, "none")], t) if D.ok(t) else TODO for a in ARCHS]
            out.append(f"{label if j == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if k < 2 else "\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def moduli_table(D):
    out = ["\\begin{tabular}{llccccccc}", "\\toprule",
           " & & & \\multicolumn{2}{c}{$K_H$} & \\multicolumn{2}{c}{$G_H$} & \\multicolumn{2}{c}{$E$} \\\\",
           "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
           "Model & Backbone & non-PD & MAE [95\\% CI] & MAPE / 10\\% / 20\\% & MAE [95\\% CI] & MAPE / 10\\% / 20\\% & "
           "MAE [95\\% CI] & MAPE / 10\\% / 20\\% \\\\", "\\midrule"]
    for a in ARCHS:
        for j, bb in enumerate(BACKBONES["elastic"]):
            r = D.mod[(a, bb)]
            cells = []
            for k in ("K_H", "G_H", "E"):
                mark = ""
                if bb != "none":
                    p = next((p for p in D.mpairs if p["modulus"] == k and p["run_a"] == r["run"]), None)
                    if p and float(p["p_holm"]) < 0.05:
                        mark = "\\textsuperscript{" + ("$\\downarrow$" if float(p["mean_diff_a_minus_b"]) < 0
                                                       else "$\\uparrow$") + "}"
                cells.append(ci(r, k, 1) + mark)
                cells.append(f"{pct(r[k + '_MAPE'])} / {pct(r[k + '_within10'], 0)} / {pct(r[k + '_within20'], 0)}")
            out.append(f"{a if j == 0 else ''} & {bb} & {r['n_nonPD']} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule")
    out.append("MatTen~\\cite{wen2024dd} & -- & -- & 7.37 (0.10) & -- & 8.38 (0.16) & -- & 20.59 (0.35) & -- \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def transfer(D):
    out = ["\\begin{tabular}{llccc}", "\\toprule", "Target & Backbone & SEGNN & SEConv & SETransformer \\\\",
           "\\midrule"]
    for k, t in enumerate(TARGETS):
        for j, bb in enumerate(BACKBONES[t]):
            cells = []
            for a in ARCHS:
                if not D.ok(t):
                    cells.append(TODO)
                    continue
                s = ci(D.m[(a, t, bb)], "MAE", DEC[t])
                if bb != "none":
                    p = D.pair(t, f"{bb} vs none", a)
                    if p and float(p["p_holm"]) < 0.05:
                        s += D.prov("\\textsuperscript{" + ("$\\downarrow$" if float(p["mean_diff_a_minus_b"]) < 0
                                                           else "$\\uparrow$") + "}")
                cells.append(s)
            label = TNAME[t] + (f" ({UNIT[t]})" if UNIT[t] else "")
            out.append(f"{label if j == 0 else ''} & {BBNAME[bb]} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if k < 2 else "\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def si_irrep(D):
    out = ["\\begin{tabular}{llccc}", "\\toprule", "Target & Irreducible component & SEGNN & SEConv & SETransformer \\\\",
           "\\midrule"]
    for k, t in enumerate(["dielectric", "piezoelectric", "elastic"]):
        for j, (key, label) in enumerate(IRREPS[t]):
            col = f"MAE_coef[{key}]"
            cells = [ci(D.m[(a, t, "none")], col, DEC[t]) if D.ok(t) else TODO for a in ARCHS]
            out.append(f"{TNAME[t] if j == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if k < 2 else "\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def si_equiv_trained(D):
    """Cells for the 'Trained' column of Table S9: largest error over all trained models of that (target, arch)."""
    lines = []
    for t in ["dielectric", "piezoelectric", "elastic"]:
        for a in ["SEConv", "SEGNN", "SETransformer"]:
            e = D.equiv.get((a, t))
            cell = (" / ".join(sci(e[op]) for op in ("rotation", "inversion", "rotoinversion"))
                    if D.ok(t) and e else TODO)
            lines.append(f"% {t} {a}\n\\expandafter\\def\\csname equiv@{t}@{a}\\endcsname{{{cell}}}")
    return "\n".join(lines)


def si_transfer(D):
    out = ["\\begin{tabular}{llccc}", "\\toprule", "Target & Backbone & SEGNN & SEConv & SETransformer \\\\",
           "\\midrule"]
    for k, t in enumerate(["dielectric", "piezoelectric", "elastic"]):
        for j, bb in enumerate(BACKBONES[t]):
            cells = []
            for a in ARCHS:
                if not D.ok(t):
                    cells.append(TODO)
                elif bb == "none":
                    cells.append(ci(D.m[(a, t, bb)], "MAE", DEC[t]))
                else:
                    p = D.pair(t, f"{bb} vs none", a)
                    cells.append(f"{signed(p['mean_diff_a_minus_b'], DEC[t])} [{signed(p['ci_lo'], DEC[t])}, "
                                 f"{signed(p['ci_hi'], DEC[t])}]; " + D.prov("$p$ " + ("" if pfmt(p['p_holm'], p['p_value']).startswith("$\\le") else "= ") + pfmt(p['p_holm'], p['p_value'])))
            out.append(f"{TNAME[t] if j == 0 else ''} & {BBNAME[bb]} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if k < 2 else "\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def si_archs(D):
    comps = [("SEGNN", "SEConv"), ("SEGNN", "SETransformer"), ("SEConv", "SETransformer")]
    out = ["\\begin{tabular}{lccc}", "\\toprule",
           "Target & SEGNN $-$ SEConv & SEGNN $-$ SETransformer & SEConv $-$ SETransformer \\\\", "\\midrule"]
    for t in ["dielectric", "piezoelectric", "elastic"]:
        cells = []
        for a1, a2 in comps:
            p = D.pair(t, f"{a1} vs {a2} (scratch)")
            cells.append(f"{signed(p['mean_diff_a_minus_b'], DEC[t])} [{signed(p['ci_lo'], DEC[t])}, "
                         f"{signed(p['ci_hi'], DEC[t])}]; " + D.prov("$p$ " + ("" if pfmt(p['p_holm'], p['p_value']).startswith("$\\le") else "= ") + pfmt(p['p_holm'], p['p_value']))
                         if D.ok(t) and p else TODO)
        out.append(f"{TNAME[t]} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def si_allmetrics(D):
    out = ["\\begin{tabular}{lllcccl}", "\\toprule",
           "Architecture & Target & Backbone & MAE [95\\% CI] & RMSE & $R^2$ & MAE per irrep \\\\", "\\midrule"]
    for i, a in enumerate(ARCHS):
        first = True
        for t in TARGETS:
            for j, bb in enumerate(BACKBONES[t]):
                if D.ok(t):
                    r = D.m[(a, t, bb)]
                    irr = ", ".join(f(r[f"MAE_coef[{key}]"], DEC[t]) for key, _ in IRREPS[t])
                    if t == "piezoelectric":
                        irr += f"; allowed {f(r['MAE_allowed'], DEC[t])}"
                    cells = [ci(r, "MAE", DEC[t]), f(r["RMSE"], DEC[t]), f(r["R2"], 3), irr]
                else:
                    cells = [TODO] * 4
                out.append(f"{a if first else ''} & {TNAME[t] if j == 0 else ''} & {bb} & " + " & ".join(cells)
                           + " \\\\")
                first = False
        out.append("\\midrule" if i < 2 else "\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def si_moduli_all(D):
    """Moduli MAE over all test materials, including those whose predicted tensor is not positive definite."""
    out = ["\\begin{tabular}{llcccc}", "\\toprule",
           "Model & Backbone & non-PD & $K_H$ & $G_H$ & $E$ \\\\", "\\midrule"]
    for i, a in enumerate(ARCHS):
        for j, bb in enumerate(BACKBONES["elastic"]):
            r = D.mod[(a, bb)]
            out.append(f"{a if j == 0 else ''} & {bb} & {r['n_nonPD']} & "
                       + " & ".join(ci(r, k + "_all", 1) for k in ("K_H", "G_H", "E")) + " \\\\")
        out.append("\\midrule" if i < 2 else "\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def si_moduli_paired(D):
    out = ["\\begin{tabular}{llccc}", "\\toprule", "Comparison & Architecture & $\\Delta K_H$ & $\\Delta G_H$ & "
           "$\\Delta E$ \\\\", "\\midrule"]
    order = [("band gap vs none", a) for a in ARCHS] + [("formation energy vs none", a) for a in ARCHS] + \
            [(f"{a1} vs {a2} (scratch)", None) for a1, a2 in [("SEGNN", "SEConv"), ("SEGNN", "SETransformer"),
                                                              ("SEConv", "SETransformer")]]
    for comp, a in order:
        cells = []
        for k in ("K_H", "G_H", "E"):
            p = next((p for p in D.mpairs if p["comparison"] == comp and p["modulus"] == k
                      and (a is None or f"R2026_{a}_-_" in p["run_a"])), None)
            cells.append(f"{signed(p['mean_diff_a_minus_b'], 2)} [{signed(p['ci_lo'], 2)}, {signed(p['ci_hi'], 2)}]; "
                         "$p$ " + ("" if pfmt(p['p_holm'], p['p_value']).startswith("$\\le") else "= ")
                         + pfmt(p['p_holm'], p['p_value']) if p else TODO)
        label = comp.replace(" (scratch)", "").replace(" vs none", " $-$ none").replace(" vs ", " $-$ ")
        out.append(f"{label} & {a or ''} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--final", nargs="+", default=[], choices=TARGETS)
    a = ap.parse_args()
    D = Data(Path(a.eval_dir), a.final)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = (f"% generated by scripts/make_tables.py from {a.eval_dir} (final: {', '.join(a.final) or 'none'}); "
            "do not edit by hand\n")
    for name, fn in [("direct", direct), ("moduli", moduli_table), ("transfer", transfer), ("si_irrep", si_irrep),
                     ("si_equiv_trained", si_equiv_trained), ("si_transfer", si_transfer), ("si_archs", si_archs),
                     ("si_allmetrics", si_allmetrics), ("si_moduli_paired", si_moduli_paired),
                     ("si_moduli_all", si_moduli_all)]:
        (out / f"{name}.tex").write_text(head + fn(D) + "\n")
        print("wrote", out / f"{name}.tex")


if __name__ == "__main__":
    main()
