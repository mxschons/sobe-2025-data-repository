# State of Brain Emulation Report 2025 - Data Visualization

This repository contains data visualization scripts for the State of Brain Emulation Report 2025.

## Repository Structure

```
sobe25-scripts/
├── scripts/
│   ├── style.py                    # Centralized style configuration
│   ├── ConnectomicsDataviz.ipynb   # Main visualization notebook
│   ├── January.ipynb               # Extended analysis
│   └── ...                         # Additional notebooks
├── data/                           # Source data files (CSV, Excel)
├── output/                         # Generated figures (SVG + PNG)
└── README.md
```

## Style System

All figures use a consistent visual style defined in `scripts/style.py`. The style includes:

### Color Palette
- **Primary Colors**: `#6B6080`, `#8B7A9E`, `#A99BBD`, `#C7BDDC` (purple tones)
- **Categorical Colors**: `#D4A84B` (gold), `#4A90A4` (teal), `#6B6080`, `#8B7355`, `#5C8B6B`
- **Accent Colors**: Gold (`#D4A84B`), Teal (`#4A90A4`), Purple (`#6B6080`)

### Typography
- Body text: Inter (fallback: Helvetica Neue, Arial)
- Colors: Text `#4A4A5A`, Titles `#3A3A48`, Captions `#787470`

### Usage
```python
from style import (
    apply_style, save_figure,
    COLORS, PRIMARY_COLORS, CATEGORICAL_COLORS,
    GOLD, TEAL, PURPLE
)

apply_style()  # Apply global matplotlib settings

# Create your figure...
fig, ax = plt.subplots()

# Save with attribution
save_figure(fig, 'my-figure')  # Saves both .svg and .png
```

## Scripts

| Notebook | Description |
|----------|-------------|
| `ConnectomicsDataviz.ipynb` | Main visualization: neuron simulations, imaging speed, compute, storage costs, neural recordings, brain initiatives, brain scans |
| `January.ipynb` | Extended analysis and visualizations |
| `information_rate.ipynb` | Parallel coordinate plots: compute/storage requirements across organisms vs computer systems |
| `Computational demands across organisms.ipynb` | Bar charts for emulation compute/storage requirements |
| `Estimated Requirements for Brain Emulation.ipynb` | Compute/storage requirement visualizations for brain emulation |
| `comparison_recording_modalities.ipynb` | Comparison of brain recording modalities |
| `Cost_estimates_Neurons.ipynb` | Cost per neuron over time analysis |
| `experiment.ipynb` | Experimental matplotlib visualizations |

## Output

All figures are saved to `output/` in both **SVG** (vector) and **PNG** (raster, 150 DPI) formats.

Each figure includes the attribution: *"Zanichelli, Schons et al, State of Brain Emulation Report 2025"*

## Dependencies

```
numpy
pandas
matplotlib
seaborn
statsmodels
openpyxl  # for Excel files
```

## Usage

1. Open any notebook in the `scripts/` directory
2. The style module is automatically imported and applied
3. Run cells to generate visualizations
4. Figures are saved to `../output/` in SVG and PNG formats
