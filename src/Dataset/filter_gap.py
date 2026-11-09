"""Drop metals and near-metals from the dielectric and piezoelectric splits (run after generate_<t>_data.py).

The electronic (clamped-ion) dielectric and piezoelectric responses of DFPT are defined only for insulators; for
(semi)metals the stored values are unphysical (e.g. eps_inf up to 2,500, NbAs e_ij = 41 C/m^2). Following the
selection of de Jong et al., Sci. Data 2, 150053 (2015), records with a Materials Project band gap below 0.1 eV
are removed. Gaps come from the MP summary endpoint and are cached in <t>/band_gaps.json. The unfiltered splits
are kept as <split>/unfiltered_{data.json,material_ids.csv}, so the script can be re-run with another threshold.

  python filter_gap.py piezo [--threshold 0.1]
  python filter_gap.py dielectric [--threshold 0.1]"""
import argparse, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("target", choices=["piezo", "dielectric"])
ap.add_argument("--threshold", type=float, default=0.1)
a = ap.parse_args()
D = Path(__file__).resolve().parent / a.target
splits = ["train", "validate", "test"]
for s in splits:  # back up once; always filter from the unfiltered copies
    for f in ["data.json", "material_ids.csv"]:
        if not (D / s / f"unfiltered_{f}").exists():
            shutil.copy2(D / s / f, D / s / f"unfiltered_{f}")
ids = {s: (D / s / "unfiltered_material_ids.csv").read_text().split("\n") for s in splits}
cache = D / "band_gaps.json"
if cache.exists():
    gaps = json.load(open(cache))
else:
    from mp_api.client import MPRester
    with MPRester((Path(__file__).resolve().parent / "APIKEY").read_text().strip()) as m:
        docs = m.materials.summary.search(material_ids=sorted({i for v in ids.values() for i in v}),
                                          fields=["material_id", "band_gap"])
    gaps = {str(d.material_id): d.band_gap for d in docs}
    json.dump(gaps, open(cache, "w"), indent=0)
report = {"threshold_eV": a.threshold, "removed": {}, "no_gap_kept": {}, "n": {}}
for s in splits:
    data = json.load(open(D / s / "unfiltered_data.json"))
    assert len(data) == len(ids[s])
    keep = [(i, d) for i, d in zip(ids[s], data) if gaps.get(i) is None or gaps[i] >= a.threshold]
    report["removed"][s] = sorted(i for i in ids[s] if gaps.get(i) is not None and gaps[i] < a.threshold)
    report["no_gap_kept"][s] = sorted(i for i in ids[s] if gaps.get(i) is None)
    report["n"][s] = {"before": len(data), "after": len(keep)}
    json.dump([d for _, d in keep], open(D / s / "data.json", "w"))
    (D / s / "material_ids.csv").write_text("\n".join(i for i, _ in keep))
json.dump(report, open(D / "gap_filter.json", "w"), indent=1)
print(a.target, {s: report["n"][s] for s in splits}, "no gap info (kept):", sum(map(len, report["no_gap_kept"].values())))
