# Analysis scripts

Scripts behind the numbers, tables and checks of the revised manuscript (AI4S-100140). The models and the
evaluation dumps are produced by the code in `src/`; these scripts only read their outputs. The trained models and the
processed data sets are attached to the GitHub release `v2.0.0`.

## Evaluation pipeline

| Step | Command | Output |
|---|---|---|
| 1. Test-set predictions | `cd src && python eval_dump.py <target> <npz_dir> <runs.txt> --probe dielectric` | one `.npz` per model (predictions, coefficients, allowed-component masks, probe embeddings) |
| 2. Metrics and tests | `python scripts/analyze.py <npz_dir> <out_dir> --n-boot 1000` | `metrics.csv`, `paired.csv`, `moduli.csv`, `paired_moduli.csv`, `relerr.csv`, `similarity.json` |
| 3. Tables | `python scripts/make_tables.py <out_dir> <tables_dir> --final elastic dielectric piezoelectric` | the LaTeX tables of the manuscript and SI |
| 4. Equal-learning-rate control | `python scripts/ctl_compare.py <npz_dir> <ctl_npz_dir> <out.csv>` | SI table of the SETransformer dielectric control |
| 5. Error structure | `python scripts/error_structure.py <npz_dir>` | error concentration and per-irrep relative errors (Results 3.2, Fig. 2(d)) |
| 6. Neighbor-shell check | `python scripts/shell_split.py src/Dataset/piezo/test <npz_dir>` | split neighbor shells and forbidden piezoelectric components (Results 3.1, Note S11) |
| 7. Moduli outliers | `python scripts/elastic_singular_check.py <npz_dir>` | non-positive-definite predicted elasticity tensors (Note S13) |

`runs.txt` lists `target name config backbone` per line; `<npz_dir>` is the output of step 1. `test_analyze.py` tests
`analyze.py` on synthetic data.

## Equivariance and data checks

| Script | What it checks |
|---|---|
| `src/equiv_test.py <target> <config> <checkpoint>` | equivariance of a trained model under rotation, inversion and rotoinversion (Note S11); `EQUIV_TEST_ID=<mp-id>` uses a test-set material instead of the random crystal |
| `src/Dataset/target_law_test.py`, `src/Dataset/piezo_old_law.py` | transformation law of the target coefficients (Note S11) |
| `src/bn_check.py <out_dir> <target> <name>...` | effect of the BatchNorm running statistics on the test error |
| `frame_check.py`, `frame_fix_test.py` | tensor/structure frame consistency of the records and whether rotating into the IEEE frame repairs it (Note S10) |
| `data_steps.py`, `data_stats.py` | records removed by each cleaning step and statistics of the processed data (Notes S7, S10) |
| `overlap.py` | overlap of the downstream test sets with the data seen by the transferred backbones |
| `model_table.py`, `model_table_pre.py` | parameter counts and irreps of the models (Table S5, S6); run from `src/`, `CKPT2024=<folder>` for the pretraining checkpoints |

Requirements: those of `src/` (`requirements.txt`); `shell_split.py`, `frame_check.py` and `data_*.py` also use pymatgen.
