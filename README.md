# State of Brain Emulation Report 2025 - Data Visualization

This repository contains data visualization scripts and a figure library website for the State of Brain Emulation Report 2025.

## Repository Structure

```
sobe25-scripts/
├── scripts/
│   ├── style.py                    # Centralized style configuration
│   ├── run_all_figures.py          # Main pipeline script
│   ├── build_downloads.py          # ZIP file generator
│   ├── ConnectomicsDataviz.ipynb   # Main visualization notebook
│   └── ...                         # Additional notebooks
├── data/                           # Source data files (CSV, Excel)
├── output/                         # Generated figures (SVG + PNG)
├── images/
│   └── hand-drawn/                 # Hand-drawn figures (add your own)
│       └── metadata.json           # Metadata for hand-drawn figures
├── website/                        # Static website pages
│   ├── figures.html                # Figure library page
│   ├── figures.js                  # Figure library JavaScript
│   ├── figures-metadata.json       # Generated figures metadata
│   ├── data.html                   # Data repository page
│   ├── data.js                     # Data repository JavaScript
│   └── data-metadata.json          # Dataset catalog metadata
├── css/                            # Stylesheets
│   ├── styles.css                  # Main design system
│   ├── figures.css                 # Figure library styles
│   └── data.css                    # Data repository styles
├── downloads/                      # Generated ZIP files
│   ├── generated-figures.zip       # All generated figures
│   └── hand-drawn-figures.zip      # All hand-drawn figures
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

### Run Full Pipeline

```bash
cd scripts
python3 run_all_figures.py
```

This generates all figures and creates downloadable ZIP files.

## Website

The `website/` folder contains static pages that can be embedded in the main report website.

### Serving Locally

```bash
# From repository root
python3 -m http.server 8000

# Then open:
# - Figure Library: http://localhost:8000/website/figures.html
# - Data Repository: http://localhost:8000/website/data.html
```

### Figure Library (`figures.html`)

Interactive gallery of all report visualizations.

**Features:**
- Filter figures by **organism** (C. elegans, Mouse, Human, etc.)
- Filter figures by **type** (Simulation, Recording, Emulation, etc.)
- Filter by **source** (Generated vs Hand-Drawn)
- Download individual figures as PNG or SVG
- Download all figures as ZIP archives

### Data Repository (`data.html`)

Browsable catalog of all validated datasets.

**Features:**
- 24 datasets organized into 8 categories
- Direct CSV download links
- Links to view/edit on GitHub
- Encourages community contributions and corrections

**Categories:**
- Neural Simulations
- Neural Recordings
- Connectomics
- Costs & Funding
- Model Organisms
- Brain Initiatives
- Resources & Repositories
- AI & Compute

### Integration

To integrate into the main website:
1. Copy HTML content or include as iframe/partial
2. Include `css/styles.css` and the page-specific CSS (`figures.css` or `data.css`)
3. Include the page-specific JavaScript
4. Ensure paths to `data/`, `output/`, `images/hand-drawn/`, and `downloads/` are correct
5. Update the GitHub URLs in `data-metadata.json` to point to your repository

## Adding Hand-Drawn Figures

1. Add your PNG and SVG files to `images/hand-drawn/`
2. Update `images/hand-drawn/metadata.json` with figure details:

```json
{
  "figures": [
    {
      "id": "synapse-diagram",
      "title": "Synapse Structure Diagram",
      "description": "Hand-drawn illustration of synaptic structure.",
      "filename": "synapse-diagram",
      "organism": ["all"],
      "type": ["hand-drawn", "anatomy"]
    }
  ]
}
```

3. Run `python3 scripts/build_downloads.py` to regenerate ZIP files

## License

All figures are licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

**Attribution:** Zanichelli, Schons et al., State of Brain Emulation Report 2025
