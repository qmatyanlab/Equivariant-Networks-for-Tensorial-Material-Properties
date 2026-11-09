# Transfer-Learning-For-Complex-Crystal-properties
A project to boost performance in predicting tensorial properties of crystals using transfer learning from elastic tensor targets due to the higher prevalence of data.

## Revision 2026-11 (AI4S-100140)

- `src/`: models (SEGNN, SEConv, SETransformer), data generation from the Materials Project (`src/Dataset/`),
  training (`run_*.py`, `configs/`), evaluation (`eval_dump.py`) and the equivariance test (`equiv_test.py`).
- `scripts/`: analysis behind the numbers, tables and checks of the revised manuscript (see `scripts/README.md`).
- Trained models and processed data sets: assets of the GitHub release `v2.0.0`
  (`models_v2.0.0.tar.gz`, `data_v2.0.0.tar.gz`). Train/validation/test material IDs are also in `src/Dataset/<target>/<split>/`.
