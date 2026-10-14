# Synthetic data

The final modeling datasets are checked in under `analysis/`. The full source and cleaned relational tables are available in [the v1.0.0 release](https://github.com/sriravi1996/panw-customer-retention-case-study/releases/tag/v1.0.0). Run `python scripts/download_data.py` to download, verify and extract them.

The archive includes all raw and cleaned CSVs, both final modeling datasets and simulator-only evaluation truth. `eval/` contains latent ground truth used for validation and must never be joined into model features. Files are entirely synthetic. Their SHA-256 checksum is pinned in the download script and accompanies the release.

Use Python 3.12 or newer for the download helper. Regeneration uses `bash run_part1.sh`; it does not require the release download.
