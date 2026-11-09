# Downloading Data
To download the data, make a file called `APIKEY` and paste in your 32 character API key from materials project. Then run the `download_data.py` file and it should do the rest, including properly sorting the data points into the same training, testing, and validation set used in the original project.

# Data pipeline of the 2026 revision (run in this folder, in this order)
1. `python generate_elastic_data.py`, `python generate_dielectric_data.py`, `python generate_piezo_data.py`:
   download, de-duplicate, align frames (`frame_align.py`), compute harmonic coefficients, split by the preset ids.
2. `python filter_gap.py dielectric` and `python filter_gap.py piezo`: remove records with an MP band gap < 0.1 eV
   (DFPT electronic responses of metals are unphysical). Unfiltered splits are kept as `unfiltered_*`;
   the removed ids are listed in `<t>/gap_filter.json`.
3. `python equivariant_stats.py`: normalization statistics from the training split, one scale per irrep and a mean
   only for 0e coefficients, written to `<t>/coef_stats.json`. The `coef_stats.json` written in step 1 (one mean and
   std per coefficient) makes the de-normalized predictions non-equivariant and must not be used.
