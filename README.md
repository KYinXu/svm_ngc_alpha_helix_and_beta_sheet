# Expanded-dataset SVM inference

This is the 12-QSAR-descriptor SVM trained on an expanded set of alpha-helix and beta-sheet antimicrobial peptides, adapted from SVM in Lee et al. (2016).

Score peptides with the checkpoint in `checkpoint/`. That folder is the only model: `svm_qsar12_model.pkl` and `svm_qsar12_zscores.csv`. The z-score file has three rows, as in the existing SVM scripts: descriptor names, means, then standard deviations. There is no model-selection flag.

## Setup

```bash
conda env create -f env.yaml
conda activate svm-ngc
```

Python is 3.10. Packages are pinned in `env.yaml` and `pyproject.toml`, including scikit-learn 1.7.2. With another environment manager, install this directory with `pip install .`.

## Run

From this directory:

```bash
python run_inference.py sanity_test/sequences.csv
python run_inference.py sanity_test/sequences.csv --windows
python run_inference.py sanity_test/sequences.csv --windows -o profiles.csv
```

Sequence mode writes `id`, `sequence`, `prediction` (`1` or `-1`), `sigma` (distance to the hyperplane; positive favors AMP), and `p_amp`.

Window mode defaults to lengths 10–35 and stride 1. Each input sequence becomes one row: mean and max residue profiles for `p_amp` and hyperplane distance, plus the best window. Sequences shorter than the minimum window are still listed, with `window_count` 0.

`--window-min-len`, `--window-max-len`, and `--stride` override those defaults.

## Input

- CSV with a `sequence` column and an id column named `id`, `peptide_id`, `sequence_id`, or `name`. Other columns are copied through.
- FASTA.
- Text with one sequence per line, or `id sequence` on each line.

Residues must be the standard 20 amino acids. Ids must be unique.

Default output is `<input directory>/results/sequence_predictions.csv` or `window_sequence_profiles.csv`.

## References

Lee, E. Y., Fulan, B. M., Wong, G. C. L., & Ferguson, A. L. (2016). Mapping membrane activity in undiscovered peptide sequence space using machine learning. *Proceedings of the National Academy of Sciences*, *113*(48), 13588–13593. [https://doi.org/10.1073/pnas.1609893113](https://doi.org/10.1073/pnas.1609893113)