# External Data Sources

This folder contains third-party datasets used in the State of Brain Emulation Report 2025. These are raw or minimally processed datasets from external sources, kept separate from our own curated data.

## Data Sources

| Folder | Source | License | Description |
|--------|--------|---------|-------------|
| `epoch-ai/` | [Epoch AI](https://epochai.org/data) | CC BY 4.0 | Notable AI models dataset with parameters, compute, and training details |
| `bosch-et-al/` | [Bosch et al.](https://github.com/cboschp/wtlandscape_mbc) | MIT | Neuroimaging technology capabilities from "The road to a whole mouse brain connectome" |

## Usage

These datasets are referenced via `paths.py`:
- `DATA_FILES["epoch_ai_models"]` - Epoch AI notable models
- `DATA_FILES["imaging_speed"]` - Bosch et al. neuroimaging speed data

## Updating External Data

When updating external datasets:
1. Download the latest version from the source
2. Place in the appropriate subfolder
3. Update the README in that subfolder with the new access date
4. Run `python3 figures/run_all_figures.py` to regenerate affected figures
5. Run `python3 validate.py` to verify data integrity

## Attribution

All external data must be properly attributed in figures using the `credit` parameter in `save_figure()`. See individual subfolder READMEs for specific attribution requirements.
