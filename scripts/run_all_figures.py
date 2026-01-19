#!/usr/bin/env python3
"""
Run all figure generation scripts for the Brain Emulation Report 2025.
This script executes the key visualizations from each notebook.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt

from style import (
    apply_style, save_figure, add_attribution,
    COLORS, PRIMARY_COLORS, CATEGORICAL_COLORS,
    GOLD, TEAL, PURPLE,
    SPECIES_NEURONS, plot_species_hlines
)

apply_style()

# Seaborn configuration
sns.set_theme(style="whitegrid")
import matplotlib as mpl
mpl.rcParams.update({
    'axes.facecolor': COLORS['plot_bg'],
    'figure.facecolor': COLORS['figure_bg'],
    'axes.edgecolor': COLORS['border'],
    'grid.color': COLORS['grid'],
    'text.color': COLORS['text'],
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
})

print("=" * 60)
print("Brain Emulation Report 2025 - Figure Generation")
print("=" * 60)

# =============================================================================
# Figure 1: Number of Neurons in Simulations
# =============================================================================
print("\n[1/10] Generating: num-neurons.svg/png")
try:
    neurons_df = pd.read_csv('../data/Neuron Simulations - TASK 3 - Sheet1.csv')
    neurons_df['Year'] = neurons_df['Simulation/Initiative'].str.extract(r'(\d{4})').apply(pd.to_datetime)

    min_year = neurons_df['Year'].min() - dt.timedelta(days=365)
    max_year = neurons_df['Year'].max() + dt.timedelta(days=365)
    label_year = dt.datetime(year=1985, month=1, day=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        neurons_df,
        x='Year',
        y='# of Neurons',
        style='Category',
        hue='Organism (random)',
        palette=CATEGORICAL_COLORS,
        s=60,
        alpha=0.8,
        ax=ax
    )
    plot_species_hlines(ax, min_year, max_year, label_year)
    ax.legend(bbox_to_anchor=(1.03, 1.03), frameon=True)
    ax.set_yscale('log')
    ax.set_xlim(min_year, max_year)
    ax.set_ylabel('Number of Neurons')
    ax.set_xlabel(None)
    ax.set_title('Neuron Simulations Over Time')
    plt.tight_layout()
    save_figure(fig, 'num-neurons')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 2: Imaging Speed
# =============================================================================
print("\n[2/10] Generating: imaging-speed.svg/png")
try:
    imaging_speed_df = pd.read_excel(
        '../data/cboschp-wtlandscape_mbc-ca8b379/0-data/maps_dates_230119.xlsx',
        skiprows=[1],
        parse_dates=['released_year'],
    )
    imaging_speed_df['imagingRate_perMachine'] = pd.to_numeric(
        imaging_speed_df['imagingRate_perMachine'], errors='coerce'
    )

    min_date = dt.date(year=1980, month=1, day=1)
    max_date = dt.date(year=2024, month=1, day=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = CATEGORICAL_COLORS

    # Subplot 1
    sns.scatterplot(
        data=imaging_speed_df, x='released_year', y='imagingRate_perMachine',
        hue='img_tech', palette=palette, s=60, alpha=0.8, ax=axes[0], legend=False,
    )
    axes[0].set_yscale('log')
    axes[0].set_title('Imaging Rate (mm³/day/machine)')
    axes[0].set_xlim(min_date, max_date)
    axes[0].set_xlabel(None)
    axes[0].set_ylabel(None)

    # Subplot 2
    sns.scatterplot(
        data=imaging_speed_df, x='released_year', y='fov_mm3',
        hue='img_tech', palette=palette, s=60, alpha=0.8, ax=axes[1], legend=False,
    )
    axes[1].set_yscale('log')
    axes[1].set_title('Volumes Imaged (mm³)')
    axes[1].set_xlim(min_date, max_date)
    axes[1].set_xlabel(None)
    axes[1].set_ylabel(None)

    # Subplot 3
    sns.scatterplot(
        data=imaging_speed_df, x='released_year', y='dsSize_TB',
        hue='img_tech', palette=palette, s=60, alpha=0.8, ax=axes[2], legend=True,
    )
    axes[2].set_yscale('log')
    axes[2].set_title('Dataset Size (TB)')
    sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1), frameon=True)
    axes[2].set_xlim(min_date, max_date)
    axes[2].set_xlabel(None)
    axes[2].set_ylabel(None)

    plt.tight_layout()
    save_figure(fig, 'imaging-speed')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 3: Compute
# =============================================================================
print("\n[3/10] Generating: compute.svg/png")
try:
    compute_df = pd.read_csv(
        '../data/ai-compute/artificial-intelligence-training-computation.csv',
        parse_dates=['Day']
    )

    species_pf = {'Human': 2000.0, 'Mouse': 10.0, 'Fly': 0.195}

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=compute_df, x='Day', y='Training computation (petaFLOP)',
        hue='Domain', palette=CATEGORICAL_COLORS, s=60, alpha=0.7, ax=ax
    )

    min_year = dt.datetime(year=1945, month=1, day=1)
    max_year = dt.datetime(year=2030, month=1, day=1)
    label_year = dt.datetime(year=1955, month=1, day=1)

    for name, val in species_pf.items():
        ax.axhline(y=val, color=COLORS['caption'], ls=':', lw=1, alpha=0.7)
        ax.text(label_year, val, f'  {name}', va='bottom', fontsize=10, color=COLORS['caption'])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=True)
    ax.set_yscale('log')
    ax.set_xlabel('Year')
    ax.set_ylabel('Training Computation (petaFLOP)')
    ax.set_xlim(min_year, max_year)
    ax.set_title('AI Training Compute vs Species Requirements')
    plt.tight_layout()
    save_figure(fig, 'compute')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 4: Storage Costs
# =============================================================================
print("\n[4/10] Generating: storage-costs.svg/png")
try:
    storage_df = pd.read_csv(
        '../data/storage-costs/historical-cost-of-computer-memory-and-storage.csv',
        parse_dates=['Year']
    )
    storage_df.rename(columns={
        'Historical price of memory': 'Memory',
        'Historical price of flash memory': 'Flash',
        'Historical price of disk drives': 'Disk',
        'Historical price of solid-state drives': 'Solid state',
    }, inplace=True)

    storage_dfl = pd.melt(
        storage_df, ['Year'],
        value_vars=['Memory', 'Flash', 'Disk', 'Solid state'],
        value_name='Cost ($ / TB)', var_name='Storage type',
    )

    species_storage_tb = {'Human': 6e3, 'Mouse': 2, 'Fruitfly': 2.5e-4, 'C. elegans': 1e-3}
    species_cost = {k: 1e6 / v for k, v in species_storage_tb.items()}

    min_year = storage_dfl['Year'].min()
    max_year = storage_dfl['Year'].max()
    label_year = dt.datetime(year=1958, month=1, day=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=storage_dfl, x='Year', y='Cost ($ / TB)',
        hue='Storage type', palette=PRIMARY_COLORS, linewidth=2.5, ax=ax
    )

    for name, val in species_cost.items():
        ax.axhline(y=val, color=COLORS['caption'], ls=':', lw=1, alpha=0.7)
        ax.text(label_year, val, f'  {name}', va='bottom', fontsize=10, color=COLORS['caption'])

    ax.set_yscale('log')
    ax.set_xlim(min_year, max_year)
    ax.set_ylabel('Cost ($ / TB)')
    ax.set_xlabel(None)
    ax.set_title('Storage Costs Over Time (Species Thresholds at $1M Budget)')
    ax.legend(frameon=True)
    plt.tight_layout()
    save_figure(fig, 'storage-costs')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 5: Neural Recordings
# =============================================================================
print("\n[5/10] Generating: neuro-recordings.svg/png")
try:
    import statsmodels.formula.api as smf

    neuro_df = pd.read_csv('../data/Neural recordings - Neurons_year.csv')

    # Fit models
    ephys_fit = smf.rlm('np.log(Neurons) ~ Year', data=neuro_df.query('Method == "Ephys"')).fit()
    imag_fit = smf.rlm('np.log(Neurons) ~ Year', data=neuro_df.query('Method == "Imaging"')).fit()

    min_year = neuro_df['Year'].min()
    max_year = neuro_df['Year'].max()
    x_lin = np.linspace(min_year, max_year, 2)
    y_lin_ephys = np.exp(ephys_fit.params['Intercept'] + ephys_fit.params['Year'] * x_lin)
    y_lin_imag = np.exp(imag_fit.params['Intercept'] + imag_fit.params['Year'] * x_lin)

    # Rename methods
    neuro_df.loc[neuro_df['Method'] == 'Ephys', 'Method'] = 'Electrophysiology'
    neuro_df.loc[neuro_df['Method'] == 'Imaging', 'Method'] = 'Fluorescence imaging'

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        neuro_df, x='Year', y='Neurons', hue='Method',
        palette=[TEAL, GOLD], s=60, alpha=0.8, ax=ax
    )

    # Regression lines
    ax.plot(x_lin, y_lin_ephys, color=COLORS['text'], ls=(0, (5, 7)), lw=2)
    ax.plot(x_lin, y_lin_imag, color=COLORS['text'], ls=(0, (5, 7)), lw=2)
    ax.plot(x_lin, y_lin_ephys, color=TEAL, ls=(6, (5, 7)), lw=2)
    ax.plot(x_lin, y_lin_imag, color=GOLD, ls=(6, (5, 7)), lw=2)

    ax.set_yscale('log')
    plot_species_hlines(ax, min_year, max_year, 1958)
    ax.set_ylabel('Simultaneously Recorded Neurons')
    ax.set_xlabel(None)
    ax.set_ylim(0.2, 1e12)
    ax.set_title('Neural Recording Capacity Over Time')
    ax.legend(frameon=True)
    plt.tight_layout()
    save_figure(fig, 'neuro-recordings')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 6: Brain Scans
# =============================================================================
print("\n[6/10] Generating: scanned-brain-tissue.svg/png")
try:
    scans_df = pd.read_csv(
        '../data/brain-scans/Copy of (best) Upwork_TASK 1 (2) - Sheet1.csv',
        parse_dates=['Year'],
    )

    min_date = dt.date(year=1975, month=1, day=1)
    max_date = dt.date(year=2025, month=1, day=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = CATEGORICAL_COLORS

    for i, (y_col, title) in enumerate([
        ('Resolution (claude)', 'Resolution (nm)'),
        ('Volume (claude)', 'Volume Imaged (μm³)'),
        ('Dataset Size (Claude)', 'Dataset Size (TB)'),
    ]):
        sns.scatterplot(
            data=scans_df, x='Year', y=y_col,
            hue='Method (claude)', style='Organism (cleaned)',
            palette=palette, s=60, alpha=0.8, ax=axes[i],
            legend=(i == 2),
        )
        axes[i].set_yscale('log')
        axes[i].set_title(title)
        axes[i].set_xlim(min_date, max_date)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel(None)
        axes[i].tick_params(axis='x', rotation=-30)

    sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1), frameon=True)
    plt.tight_layout()
    save_figure(fig, 'scanned-brain-tissue')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 7: Recording Modalities Comparison
# =============================================================================
print("\n[7/10] Generating: recording-modalities.svg/png")
try:
    categories = ["Resolution", "Speed", "Duration", "Volume"]
    x = np.arange(len(categories))

    methods = {
        "fUS": {"values": [3, 3, 4, 3], "color": CATEGORICAL_COLORS[0], "ls": "-", "marker": "o"},
        "Calcium Imaging": {"values": [4, 2, 2, 2], "color": CATEGORICAL_COLORS[1], "ls": "-", "marker": "o"},
        "Voltage Imaging": {"values": [4, 3.25, 1.5, 1.75], "color": CATEGORICAL_COLORS[2], "ls": "-", "marker": "o"},
        "MEA": {"values": [4, 4, 4, 1.5], "color": GOLD, "ls": "-", "marker": "o"},
        "EEG": {"values": [1, 4, 3, 3.5], "color": CATEGORICAL_COLORS[3], "ls": "--", "marker": "o"},
        "fMRI": {"values": [2, 1, 3, 4], "color": CATEGORICAL_COLORS[4], "ls": "--", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(-0.2, len(categories) - 1 + 0.2)
    ax.set_ylim(0.8, 4.2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight="500")
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    for xi in x:
        ax.vlines(xi, 1, 4, colors=COLORS['grid'], linewidth=1, zorder=0)
    for y in [1, 2, 3, 4]:
        ax.hlines(y, x[0], x[-1], colors=COLORS['grid'], linewidth=1, linestyles="--", zorder=0)

    axis_labels = {
        "Resolution": ["Millions", "10s of thousands", "Thousands", "Single Cell"],
        "Speed": ["1 Hz", "10 Hz", "100 Hz", "1000 Hz"],
        "Duration": ["Seconds", "Minutes", "Hours", "Days"],
        "Volume": ["1 µm³", "1 mm³", "1 cm³", "1 dm³"],
    }

    for i, cat in enumerate(categories):
        for y, label in zip([1, 2, 3, 4], axis_labels[cat]):
            ha = "left" if cat == "Volume" else "right"
            x_offset = 0.03 if cat == "Volume" else -0.03
            ax.text(i + x_offset, y, label, ha=ha, va="center", fontsize=9, color=COLORS['text'])

    for name, props in methods.items():
        ax.plot(x, props["values"], linestyle=props["ls"], marker=props["marker"],
                linewidth=2.5, markersize=6, color=props["color"],
                markeredgecolor='white', markeredgewidth=1.5, label=name)

    ax.set_title("Brain Tissue Recording Modalities Comparison", fontsize=14, pad=20)
    ax.legend(title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    save_figure(fig, 'recording-modalities')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 8: Emulation Requirements
# =============================================================================
print("\n[8/10] Generating: emulation-*.svg/png")
try:
    categories_organisms = ['C. elegans', 'Fly', 'Mouse', 'Human']
    categories_full = categories_organisms + ['H100', 'xAI Colossus']
    x_pos_full = np.arange(len(categories_full))
    x_pos_org = np.arange(len(categories_organisms))

    # Time-based compute
    compute_mids = np.array([1.87e9, 3.76e12, 8.88e15, 1.12e19])
    compute_errs = np.array([8.25e8, 1.05e12, 2.13e15, 2.7e18])
    ref_compute = [np.nan, np.nan, np.nan, np.nan, 1E+15, 1E+20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos_org, compute_mids, yerr=compute_errs, alpha=0.85, color=GOLD,
           ecolor=COLORS['text'], capsize=5, label='Organisms', edgecolor='white')
    ax.bar(x_pos_full, ref_compute, alpha=0.7, color=COLORS['caption'],
           label='Reference', edgecolor='white')
    ax.set_yscale('log')
    ax.set_xticks(x_pos_full)
    ax.set_xticklabels(categories_full, rotation=30, ha='right')
    ax.set_ylabel('Compute (FLOP/s)')
    ax.set_title('Emulation Compute Requirements (Time-Based)')
    ax.legend(frameon=True)
    ax.set_ylim(1e8, 1e21)
    plt.tight_layout()
    save_figure(fig, 'emulation-compute-time-based')
    plt.close()

    # Storage
    storage_mids = np.array([2.6e5, 6.54e8, 1.63e12, 2.05e15])
    storage_errs = np.array([9.4e4, 2.21e8, 5.45e11, 6.85e14])
    ref_storage = [np.nan, np.nan, np.nan, np.nan, 8E+10, 1E+16]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos_org, storage_mids, yerr=storage_errs, alpha=0.85, color=TEAL,
           ecolor=COLORS['text'], capsize=5, label='Organisms', edgecolor='white')
    ax.bar(x_pos_full, ref_storage, alpha=0.7, color=COLORS['caption'],
           label='Reference', edgecolor='white')
    ax.set_yscale('log')
    ax.set_xticks(x_pos_full)
    ax.set_xticklabels(categories_full, rotation=30, ha='right')
    ax.set_ylabel('Storage (Bytes)')
    ax.set_title('Emulation Storage Requirements')
    ax.legend(frameon=True)
    ax.set_ylim(1e5, 1e17)
    plt.tight_layout()
    save_figure(fig, 'emulation-storage-requirements')
    plt.close()

    # Event-driven compute
    event_mids = np.array([5.27e8, 2.61e11, 1.69e14, 6.04e16])
    event_errs = np.array([5.23e8, 2.5e11, 1.42e14, 2.64e16])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos_org, event_mids, yerr=event_errs, alpha=0.85, color=PURPLE,
           ecolor=COLORS['text'], capsize=5, label='Organisms', edgecolor='white')
    ax.bar(x_pos_full, ref_compute, alpha=0.7, color=COLORS['caption'],
           label='Reference', edgecolor='white')
    ax.set_yscale('log')
    ax.set_xticks(x_pos_full)
    ax.set_xticklabels(categories_full, rotation=30, ha='right')
    ax.set_ylabel('Compute (FLOP/s)')
    ax.set_title('Emulation Compute Requirements (Event-Driven)')
    ax.legend(frameon=True)
    ax.set_ylim(1e6, 1e21)
    plt.tight_layout()
    save_figure(fig, 'emulation-compute-event-driven')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 9: Cost per Neuron
# =============================================================================
print("\n[9/10] Generating: cost-per-neuron.svg/png")
try:
    df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Cost estimates Neuron Reconstruction.csv')
    df['CostPerNeuron'] = df['Cost / Neuron'].replace('[\\$,]', '', regex=True).astype(float)

    def make_label(row):
        if row['Organism'].strip().lower().startswith('mouse'):
            return "Brain Connects Mouse Prototype (2024)"
        return f"{row['Organism'].strip()} ({int(row['Year'])})"
    df['PlotLabel'] = df.apply(make_label, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = CATEGORICAL_COLORS + [GOLD]
    for idx, (year, cost, label) in enumerate(zip(df['Year'], df['CostPerNeuron'], df['PlotLabel'])):
        ax.scatter(year, cost, s=80, color=colors[idx % len(colors)], label=label,
                   edgecolor='white', linewidth=1.5, alpha=0.9, zorder=3)

    ax.axhline(10, linestyle='--', color=COLORS['caption'], lw=1.5)
    ax.text(2050, 14, 'Mouse connectome for $1B', ha='right', fontsize=10, color=COLORS['caption'])
    ax.axhline(0.01, linestyle='--', color=COLORS['caption'], lw=1.5)
    ax.text(2050, 0.02, 'Human connectome for $1B', ha='right', fontsize=10, color=COLORS['caption'])

    ax.set_yscale('log')
    ax.set_xlim(1980, 2050)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cost per Neuron (USD)')
    ax.set_title('Cost per Neuron Over Time')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    save_figure(fig, 'cost-per-neuron')
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 10: Initiatives
# =============================================================================
print("\n[10/10] Generating: initiatives*.svg/png")
try:
    from matplotlib.patches import Patch

    brain_proj_df = pd.read_csv(
        '../data/initiatives/Overview of Brain Initiatives T4 v2.xlsx - Sheet1.csv',
        parse_dates=['Start Year (cleaned)', 'End Year (cleaned)']
    )
    brain_proj_df.dropna(subset=['Start Year (cleaned)', 'Budget (in million $) (cleaned)'], inplace=True)
    brain_proj_df['Category'] = 'Brain'
    brain_proj_df['End Year (cleaned)'] = brain_proj_df['End Year (cleaned)'].fillna(dt.datetime(2024, 12, 31))

    other_proj_df = pd.read_csv(
        '../data/initiatives/Digital Human Intelligence Figures - Costs of different projects.csv',
        parse_dates=['StartYear', 'EndYear'],
        converters={'Adjusted2024_M': lambda s: 1e3 * float(s.replace('$', ''))}
    )

    all_proj_df = pd.concat([
        other_proj_df[['Name', 'Adjusted2024_M', 'Category', 'StartYear', 'EndYear']].rename(
            columns={'Adjusted2024_M': 'Budget_M'}),
        brain_proj_df[['Project name', 'Category', 'Budget (in million $) (cleaned)',
                       'Start Year (cleaned)', 'End Year (cleaned)']].rename(
            columns={'Project name': 'Name', 'Budget (in million $) (cleaned)': 'Budget_M',
                     'Start Year (cleaned)': 'StartYear', 'End Year (cleaned)': 'EndYear'}),
    ], ignore_index=True)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(all_proj_df, x='StartYear', y='Budget_M', hue='Category',
                    palette=CATEGORICAL_COLORS + PRIMARY_COLORS[:4], s=80, alpha=0.8, ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=True)
    ax.set_yscale('log')
    ax.set_ylabel('Budget (Million $)')
    ax.set_xlabel(None)
    ax.set_title('Megaproject Budgets by Start Year')
    plt.tight_layout()
    save_figure(fig, 'initiatives1')
    plt.close()

    # Compute project durations and midpoints for initiatives2, 4, 5
    proj_durations = all_proj_df['EndYear'] - all_proj_df['StartYear']
    proj_midpoints = all_proj_df['StartYear'] + proj_durations / 2

    proj_categories = all_proj_df['Category'].unique()
    all_colors = CATEGORICAL_COLORS + PRIMARY_COLORS
    category_colors = [all_colors[i % len(all_colors)] for i in range(len(proj_categories))]
    category_colormap = dict(zip(proj_categories, category_colors))
    proj_colors = [category_colormap[cat] for cat in all_proj_df['Category']]

    # initiatives2 - Megaproject Budgets and Durations
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(all_proj_df)):
        ax.errorbar(
            [proj_midpoints[i]],
            [1e6 * all_proj_df.loc[i, 'Budget_M']],
            ls='none',
            xerr=[proj_durations[i] / 2],
            capsize=4,
            ecolor=proj_colors[i],
            alpha=0.8,
        )
    ax.set_yscale('log')
    ax.set_xlim(dt.datetime(year=1940, month=1, day=1), dt.datetime(year=2035, month=1, day=1))
    ax.set_ylabel('Budget ($)')
    ax.set_xlabel(None)
    legend_handles = [
        Patch(facecolor=color, edgecolor=COLORS['border'], label=category)
        for category, color in zip(proj_categories, category_colors)
    ]
    for i, proj in all_proj_df.head(6).iterrows():
        ax.text(proj_midpoints[i], 1e6 * proj['Budget_M'], proj['Name'],
                ha='center', va='bottom', fontsize=9, color=COLORS['text'])
    ax.legend(handles=legend_handles, title="Category", loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.set_title('Megaproject Budgets and Durations')
    plt.tight_layout()
    save_figure(fig, 'initiatives2')
    plt.close()

    # initiatives3 - KDE plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(all_proj_df, x='Budget_M', label='All projects', fill=True, log_scale=True,
                color=COLORS['grid'], alpha=0.5, ax=ax)
    sns.kdeplot(all_proj_df.query('Category == "Brain"'), x='Budget_M', label='Brain projects',
                fill=True, log_scale=True, color=GOLD, alpha=0.7, ax=ax)
    ax.set_title('Brain Project Budgets vs All Megaprojects')
    ax.set_xlabel('Budget (Million $)')
    ax.legend(frameon=True)
    plt.tight_layout()
    save_figure(fig, 'initiatives3')
    plt.close()

    # initiatives4 - Budgets with KDE overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(all_proj_df)):
        ax.errorbar(
            [proj_midpoints[i]],
            [all_proj_df.loc[i, 'Budget_M']],
            ls='none',
            xerr=[proj_durations[i] / 2],
            capsize=4,
            ecolor=proj_colors[i],
            alpha=0.8,
        )
    ax.set_yscale('log')
    ax.set_xlim(dt.datetime(year=1940, month=1, day=1), dt.datetime(year=2035, month=1, day=1))
    ax.set_ylabel('Budget (Million $)')
    for i, proj in all_proj_df.head(6).iterrows():
        ax.text(proj_midpoints[i], proj['Budget_M'], proj['Name'],
                ha='center', va='bottom', fontsize=9, color=COLORS['text'])
    ax.legend(handles=legend_handles, title="Category", loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.set_title('Megaproject Budgets')
    ax2 = ax.twiny()
    ax2.set_xticks([])
    ax2.set_xlim(0, 1.5)
    sns.kdeplot(all_proj_df.query('Category == "Brain"'), y='Budget_M', label='Brain', color=GOLD, fill=True, log_scale=True, ax=ax2, alpha=0.5)
    sns.kdeplot(all_proj_df.query('Category != "Brain"'), y='Budget_M', label='Other', color=COLORS['caption'], fill=True, log_scale=True, ax=ax2, alpha=0.3)
    ax2.legend(loc=(0.3, 0.3), title='Distributions', frameon=True)
    ax.set_ylim(1e0, 5e6)
    plt.tight_layout()
    save_figure(fig, 'initiatives4')
    plt.close()

    # initiatives5 - Budget Distributions by Category Over Time
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        all_proj_df,
        x='StartYear',
        y='Budget_M',
        hue='Category',
        palette=category_colormap,
        log_scale=(False, True),
        fill=True,
        alpha=0.4,
        warn_singular=False,
        ax=ax
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=True)
    ax.set_ylim(1e0, 5e6)
    ax.set_xlim(dt.datetime(year=1950, month=1, day=1), dt.datetime(year=2035, month=1, day=1))
    ax.set_ylabel('Budget (Million $)')
    ax.set_xlabel(None)
    ax.set_title('Budget Distributions by Category Over Time')
    plt.tight_layout()
    save_figure(fig, 'initiatives5')
    plt.close()

    # initiatives6 - Budget Distributions by Category
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(all_proj_df, x='Budget_M', label='All projects', log_scale=True, fill=True, color=COLORS['grid'], alpha=0.5, ax=ax)
    for i, category in enumerate(proj_categories):
        color = category_colors[i]
        sns.kdeplot(all_proj_df.query(f'Category == "{category}"'), x='Budget_M', label=category, fill=False, log_scale=True, color=color, ax=ax)
    ax.set_title('Budget Distributions by Category')
    ax.set_xlabel('Budget (Million $)')
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    save_figure(fig, 'initiatives6')
    plt.close()

    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 11: Simulation Heatmap
# =============================================================================
print("\n[11/13] Generating: sim-heatmap.svg/png")
try:
    import textwrap
    import os

    # Load simulation data
    neuro_sim_df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Computational Models of the Brain.csv')

    # Define organisms and data columns
    organisms = ['C. elegans', 'Drosophila', 'Zebrafish', 'Mouse', 'Human']
    # Map organism names
    organism_map = {
        'Mammalian': 'Mouse',
        'Silicon': None,  # Skip
    }
    neuro_sim_df['Organism_mapped'] = neuro_sim_df['Organism'].replace(organism_map)
    neuro_sim_df = neuro_sim_df[neuro_sim_df['Organism_mapped'].isin(organisms)]

    data_columns = [
        'Connectivity accuracy',
        'Percentage of neurons',
        'Neurontypes',
        'Plasticity',
        'Functional Accuracy',
        'Neuromodulation',
        'Temporal resolution',
        'Behavior',
        'Personality-defining Characteristics',
        'Learning',
    ]

    # Drop rows with missing data
    neuro_sim_df = neuro_sim_df.dropna(subset=data_columns, how='all')

    # Convert to numeric
    for col in data_columns:
        neuro_sim_df[col] = pd.to_numeric(neuro_sim_df[col], errors='coerce')

    neuro_sim_df = neuro_sim_df.dropna(subset=data_columns)

    if len(neuro_sim_df) > 0:
        fig, sim_ax = plt.subplots(figsize=(8, 8))
        shrink = 0.9

        heatmap = sns.heatmap(
            neuro_sim_df[data_columns].astype(float),
            ax=sim_ax,
            cbar=True,
            cmap='Blues',
            cbar_kws=dict(
                orientation="vertical",
                location="left",
                shrink=shrink,
                aspect=20*shrink,
                pad=0.07,
            ),
        )

        cbar = heatmap.collections[0].colorbar
        cbar.set_ticks([])
        cbar.ax.text(0, -0.02, "Worse", transform=cbar.ax.transAxes, ha='left', va='top')
        cbar.ax.text(0, 1.02, "Better", transform=cbar.ax.transAxes, ha='left', va='bottom')

        sim_ax.yaxis.tick_right()
        sim_ax.set_xticks(
            np.arange(len(data_columns)) + 0.3,
            [textwrap.fill(col, width=25) for col in data_columns],
            rotation=-45,
            ha='left',
        )
        sim_ax.set_yticks(
            np.arange(len(neuro_sim_df)) + 0.5,
            labels=[
                f'{row["First Author"]} ({int(row["Year"])})'
                for _, row in neuro_sim_df.iterrows()
            ],
            rotation=0,
        )
        sim_ax.tick_params(axis='both', which='both', length=0)
        for direction in ['bottom', 'right']:
            sim_ax.spines[direction].set_visible(True)
            sim_ax.spines[direction].set_color(COLORS['border'])
        plt.grid()
        plt.tight_layout()
        save_figure(fig, 'sim-heatmap')
        plt.close()
        print("   Done!")
    else:
        print("   Skipped - no valid data")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 12: Recording Heatmap
# =============================================================================
print("\n[12/13] Generating: rec-heatmap.svg/png")
try:
    import textwrap

    # Load recording data
    neuro_rec_df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Neural Dynamics References.csv')

    # Rename columns to match expected format
    neuro_rec_df = neuro_rec_df.rename(columns={
        'First author': 'First Author',
        'Temporal resolution: Hz': 'Temporal resolution',
        'Duration single session min per Individual': 'Duration',
        'Resolution in µm isotropic': 'Resolution',
        'Number of neurons': 'Neurons',
    })

    rec_data_columns = [
        'Temporal resolution',
        'Duration',
        'Resolution',
        'Neurons',
    ]

    # Convert to numeric
    for col in rec_data_columns:
        if col in neuro_rec_df.columns:
            neuro_rec_df[col] = pd.to_numeric(neuro_rec_df[col], errors='coerce')

    neuro_rec_df = neuro_rec_df.dropna(subset=rec_data_columns)

    if len(neuro_rec_df) > 0:
        # Log-scale the data for visualization
        rec_df_scaled = neuro_rec_df[rec_data_columns].apply(lambda x: np.log10(x.replace(0, np.nan)))

        fig, rec_ax = plt.subplots(figsize=(8, 8))
        shrink = 0.9

        heatmap = sns.heatmap(
            rec_df_scaled,
            ax=rec_ax,
            cbar=True,
            cmap='OrRd',
            cbar_kws=dict(
                orientation="vertical",
                location="left",
                shrink=shrink,
                aspect=20*shrink,
                pad=0.12,
                label='log10 scale',
            ),
        )

        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.tick_right()

        rec_ax.yaxis.tick_right()
        rec_ax.set_xticks(
            np.arange(len(rec_data_columns)) + 0.5,
            [textwrap.fill(col, width=12) for col in rec_data_columns],
        )
        rec_ax.set_yticks(
            np.arange(len(neuro_rec_df)) + 0.5,
            labels=[
                f'{row["First Author"]} ({int(row["Year"])})'
                for _, row in neuro_rec_df.iterrows()
            ],
            rotation=0,
        )
        rec_ax.tick_params(axis='both', which='both', length=0)
        for direction in ['bottom', 'right']:
            rec_ax.spines[direction].set_visible(True)
            rec_ax.spines[direction].set_color(COLORS['border'])
        plt.grid()
        plt.tight_layout()
        save_figure(fig, 'rec-heatmap')
        plt.close()
        print("   Done!")
    else:
        print("   Skipped - no valid data")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 13: Neuro-sim radar charts (individual organisms) - WITH TICKS AND INFO BOXES
# =============================================================================
print("\n[13/13] Generating: neuro-sim/*.svg/png")
try:
    import textwrap
    import os

    # Create output directory if it doesn't exist
    os.makedirs('../output/neuro-sim', exist_ok=True)

    # Load simulation data
    neuro_sim_df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Computational Models of the Brain.csv')

    organisms = ['C. elegans', 'Drosophila', 'Zebrafish', 'Mouse', 'Human']
    organism_map = {
        'Mammalian': 'Mouse',
        'Silicon': None,
    }
    neuro_sim_df['Organism_mapped'] = neuro_sim_df['Organism'].replace(organism_map)

    data_columns = [
        'Connectivity accuracy',
        'Percentage of neurons',
        'Neurontypes',
        'Plasticity',
        'Functional Accuracy',
        'Neuromodulation',
        'Temporal resolution',
        'Behavior',
        'Personality-defining Characteristics',
        'Learning',
    ]

    # Short labels for axes
    column_labels = [
        'Connectivity\nAccuracy',
        '% of\nNeurons',
        'Neuron\nTypes',
        'Plasticity',
        'Functional\nAccuracy',
        'Neuro-\nmodulation',
        'Temporal\nResolution',
        'Behavior',
        'Personality',
        'Learning',
    ]

    # Convert to numeric
    for col in data_columns:
        neuro_sim_df[col] = pd.to_numeric(neuro_sim_df[col], errors='coerce')

    for organism in organisms:
        organism_df = neuro_sim_df[neuro_sim_df['Organism_mapped'] == organism]
        organism_df = organism_df.dropna(subset=data_columns)

        if organism_df.empty:
            print(f"   No data for {organism}, skipping.")
            continue

        N = len(data_columns)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        width = 2 * np.pi / N

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Set radial limits (0-3 scale for simulation data)
        ax.set_ylim(0, 3.5)

        # Draw concentric circles for tick marks
        tick_values = [0, 1, 2, 3]
        tick_labels = ['0', '1', '2', '3']
        for tick_val in tick_values:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta, [tick_val]*100, color=COLORS['grid'], lw=0.5, alpha=0.7)

        # Draw radial dividers
        for i in range(N):
            theta_div = 2*np.pi * (i-1/2) / N
            ax.plot([theta_div, theta_div], [0, 3.5], color=COLORS['grid'], lw=0.5, alpha=0.7)

        # Add tick labels in the middle (at angle 0)
        for tick_val, tick_label in zip(tick_values[1:], tick_labels[1:]):
            ax.text(0, tick_val, tick_label, ha='center', va='center',
                    fontsize=9, color=COLORS['caption'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

        num_simulations = len(organism_df)
        base_alpha = 0.7
        normalized_alpha = 1 - (1 - base_alpha)**(1/max(num_simulations, 1))

        # Compute max values per column for info boxes
        max_values = organism_df[data_columns].max()

        for _, row in organism_df.iterrows():
            try:
                values = row[data_columns].astype(float).values
            except ValueError:
                continue

            ax.bar(
                x=angles,
                height=values,
                width=width,
                bottom=0.0,
                alpha=normalized_alpha,
                color=TEAL,
                edgecolor='none'
            )

        # Add axis labels
        ax.set_xticks(angles)
        ax.set_xticklabels(column_labels, fontsize=10, color=COLORS['text'])
        ax.tick_params(axis='x', pad=15)

        # Add info boxes at outside showing max values
        for i, (col, label) in enumerate(zip(data_columns, column_labels)):
            angle = angles[i]
            max_val = max_values[col]
            if pd.notna(max_val):
                # Position outside the chart
                r_pos = 3.8
                x_pos = r_pos * np.cos(np.pi/2 - angle)
                y_pos = r_pos * np.sin(np.pi/2 - angle)

                # Format the value
                if max_val >= 1:
                    val_str = f"{max_val:.1f}"
                else:
                    val_str = f"{max_val:.2f}"

                # Add info box
                ax.annotate(
                    f"Max: {val_str}",
                    xy=(angle, 3.2),
                    xytext=(angle, 4.0),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color=COLORS['text'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['grid'], edgecolor=COLORS['border'], alpha=0.9),
                )

        ax.set_yticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)

        # Add title
        ax.set_title(f'{organism} - Simulation Characteristics', fontsize=14, pad=60, color=COLORS['title'])

        plt.tight_layout()
        add_attribution(fig)
        fig.savefig(f'../output/neuro-sim/{organism}.svg', format='svg', bbox_inches='tight', pad_inches=0.2)
        fig.savefig(f'../output/neuro-sim/{organism}.png', format='png', dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 14: Neuro-rec radar charts - PER ORGANISM with ticks and info boxes
# =============================================================================
print("\n[14/16] Generating: neuro-rec/*.svg/png")
try:
    import textwrap
    import os

    # Create output directory if it doesn't exist
    os.makedirs('../output/neuro-rec', exist_ok=True)

    # Load recording data
    neuro_rec_df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Neural Dynamics References.csv')

    # Rename columns for convenience
    neuro_rec_df = neuro_rec_df.rename(columns={
        'First author': 'First Author',
        'Temporal resolution: Hz': 'Temporal_resolution',
        'Duration single session min per Individual': 'Duration',
        'Resolution in µm isotropic': 'Resolution',
        'Number of neurons': 'Neurons',
        'Fixated / moving': 'FixMov',
    })

    organisms = ['C. elegans', 'Drosophila', 'Zebrafish', 'Mouse', 'Human']
    # Standardize organism names
    neuro_rec_df['Organism'] = neuro_rec_df['Organism'].replace({
        'C. Elegans': 'C. elegans',
        'Zebrafish Larvae': 'Zebrafish',
    })

    rec_data_columns = ['Temporal_resolution', 'Duration', 'Resolution', 'Neurons']
    rec_column_labels = ['Temporal\nResolution (Hz)', 'Duration\n(min)', 'Resolution\n(µm)', 'Number of\nNeurons']
    rec_units = ['Hz', 'min', 'µm', 'neurons']

    # Convert to numeric
    for col in rec_data_columns:
        if col in neuro_rec_df.columns:
            neuro_rec_df[col] = pd.to_numeric(neuro_rec_df[col], errors='coerce')

    # Define log scale parameters for normalization
    # These define the range mapping: log10(value) maps to radial position
    log_ranges = {
        'Temporal_resolution': (-1, 3),  # 0.1 Hz to 1000 Hz
        'Duration': (-1, 4),              # 0.1 min to 10000 min
        'Resolution': (-1, 4),            # 0.1 µm to 10000 µm
        'Neurons': (0, 11),               # 1 to 100 billion
    }

    def format_value(val, col):
        """Format value with scientific notation if needed."""
        if pd.isna(val) or val == 0:
            return "N/A"
        if val >= 1e6:
            exp = int(np.floor(np.log10(val)))
            mantissa = val / 10**exp
            return f"{mantissa:.1f} × 10^{exp}"
        elif val >= 1000:
            return f"{val:.0f}"
        elif val >= 1:
            return f"{val:.1f}"
        else:
            return f"{val:.2f}"

    def normalize_log_value(val, col):
        """Normalize value to 0-5 scale using log transformation."""
        if pd.isna(val) or val <= 0:
            return 0
        log_val = np.log10(val)
        log_min, log_max = log_ranges[col]
        # Normalize to 0-5 range
        normalized = 5 * (log_val - log_min) / (log_max - log_min)
        return max(0, min(5, normalized))

    # Generate per-organism radar charts
    for organism in organisms:
        organism_df = neuro_rec_df[neuro_rec_df['Organism'] == organism]
        organism_df = organism_df.dropna(subset=rec_data_columns, how='all')

        # Filter to rows that have at least some data
        organism_df = organism_df[organism_df[rec_data_columns].notna().any(axis=1)]

        if organism_df.empty:
            print(f"   No recording data for {organism}, skipping.")
            continue

        N = len(rec_data_columns)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        width = 2 * np.pi / N

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Set radial limits
        ax.set_ylim(0, 6)

        # Draw concentric circles for tick marks (log scale: 10^0, 10^1, 10^2, etc.)
        tick_positions = [1, 2, 3, 4, 5]
        for tick_pos in tick_positions:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta, [tick_pos]*100, color=COLORS['grid'], lw=0.5, alpha=0.7)

        # Draw radial dividers
        for i in range(N):
            theta_div = 2*np.pi * (i-1/2) / N
            ax.plot([theta_div, theta_div], [0, 6], color=COLORS['grid'], lw=0.5, alpha=0.7)

        # Add tick labels in the middle (at angle 0) - showing log scale
        tick_labels_log = ['10^0', '10^1', '10^2', '10^3', '10^4']
        for tick_pos, tick_label in zip(tick_positions, tick_labels_log):
            ax.text(0, tick_pos, tick_label, ha='center', va='center',
                    fontsize=9, color=COLORS['caption'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

        num_recs = len(organism_df)
        base_alpha = 0.7
        normalized_alpha = 1 - (1 - base_alpha)**(1/max(num_recs, 1))

        # Compute max values per column for info boxes
        max_values = {}
        for col in rec_data_columns:
            col_data = organism_df[col].dropna()
            if len(col_data) > 0:
                max_values[col] = col_data.max()
            else:
                max_values[col] = np.nan

        # Plot data
        for _, row in organism_df.iterrows():
            values = []
            for col in rec_data_columns:
                val = row[col]
                normalized = normalize_log_value(val, col)
                values.append(normalized)

            if any(v > 0 for v in values):
                ax.bar(
                    x=angles,
                    height=values,
                    width=width,
                    bottom=0.0,
                    alpha=normalized_alpha,
                    color=GOLD,
                    edgecolor='none'
                )

        # Add axis labels
        ax.set_xticks(angles)
        ax.set_xticklabels(rec_column_labels, fontsize=10, color=COLORS['text'])
        ax.tick_params(axis='x', pad=15)

        # Add info boxes at outside showing max values
        for i, (col, label, unit) in enumerate(zip(rec_data_columns, rec_column_labels, rec_units)):
            angle = angles[i]
            max_val = max_values.get(col, np.nan)
            if pd.notna(max_val):
                val_str = format_value(max_val, col)
                # Add info box
                ax.annotate(
                    f"Max: {val_str} {unit}",
                    xy=(angle, 5.2),
                    xytext=(angle, 5.8),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color=COLORS['text'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['grid'], edgecolor=COLORS['border'], alpha=0.9),
                )

        ax.set_yticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)

        # Add title
        ax.set_title(f'{organism} - Recording Characteristics', fontsize=14, pad=60, color=COLORS['title'])

        plt.tight_layout()
        add_attribution(fig)
        fig.savefig(f'../output/neuro-rec/{organism}.svg', format='svg', bbox_inches='tight', pad_inches=0.2)
        fig.savefig(f'../output/neuro-rec/{organism}.png', format='png', dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # Also generate fixated/moving aggregated charts
    for fixmov in ['fixated', 'moving']:
        sub_df = neuro_rec_df[neuro_rec_df['FixMov'] == fixmov]
        sub_df = sub_df.dropna(subset=rec_data_columns, how='all')
        sub_df = sub_df[sub_df[rec_data_columns].notna().any(axis=1)]

        if sub_df.empty:
            print(f"   No data for {fixmov}, skipping.")
            continue

        N = len(rec_data_columns)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        width = 2 * np.pi / N

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.set_ylim(0, 6)

        # Draw concentric circles
        for tick_pos in tick_positions:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta, [tick_pos]*100, color=COLORS['grid'], lw=0.5, alpha=0.7)

        # Draw radial dividers
        for i in range(N):
            theta_div = 2*np.pi * (i-1/2) / N
            ax.plot([theta_div, theta_div], [0, 6], color=COLORS['grid'], lw=0.5, alpha=0.7)

        # Add tick labels
        for tick_pos, tick_label in zip(tick_positions, tick_labels_log):
            ax.text(0, tick_pos, tick_label, ha='center', va='center',
                    fontsize=9, color=COLORS['caption'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

        num_recs = len(sub_df)
        normalized_alpha = 1 - (1 - 0.7)**(1/max(num_recs, 1))

        # Compute max values
        max_values = {}
        for col in rec_data_columns:
            col_data = sub_df[col].dropna()
            if len(col_data) > 0:
                max_values[col] = col_data.max()
            else:
                max_values[col] = np.nan

        for _, row in sub_df.iterrows():
            values = [normalize_log_value(row[col], col) for col in rec_data_columns]
            if any(v > 0 for v in values):
                ax.bar(x=angles, height=values, width=width, bottom=0.0,
                       alpha=normalized_alpha, color=GOLD, edgecolor='none')

        ax.set_xticks(angles)
        ax.set_xticklabels(rec_column_labels, fontsize=10, color=COLORS['text'])
        ax.tick_params(axis='x', pad=15)

        # Add info boxes
        for i, (col, label, unit) in enumerate(zip(rec_data_columns, rec_column_labels, rec_units)):
            angle = angles[i]
            max_val = max_values.get(col, np.nan)
            if pd.notna(max_val):
                val_str = format_value(max_val, col)
                ax.annotate(
                    f"Max: {val_str} {unit}",
                    xy=(angle, 5.2),
                    xytext=(angle, 5.8),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color=COLORS['text'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['grid'], edgecolor=COLORS['border'], alpha=0.9),
                )

        ax.set_yticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_title(f'Neural Recordings ({fixmov.capitalize()})', fontsize=14, pad=60, color=COLORS['title'])

        plt.tight_layout()
        add_attribution(fig)
        fig.savefig(f'../output/neuro-rec/{fixmov}.svg', format='svg', bbox_inches='tight', pad_inches=0.2)
        fig.savefig(f'../output/neuro-rec/{fixmov}.png', format='png', dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 15: All sim-rec combined grid
# =============================================================================
print("\n[15/16] Generating: all-sim-rec.svg/png")
try:
    import textwrap

    # Load both datasets
    neuro_sim_df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Computational Models of the Brain.csv')
    neuro_rec_df = pd.read_csv('../data/State of Brain Emulation Report 2025 Data Repository - Neural Dynamics References.csv')

    organisms = ['C. elegans', 'Drosophila', 'Zebrafish', 'Mouse', 'Human']

    # Standardize organism names in sim data
    organism_map = {'Mammalian': 'Mouse', 'Silicon': None}
    neuro_sim_df['Organism_mapped'] = neuro_sim_df['Organism'].replace(organism_map)

    # Standardize organism names in rec data
    neuro_rec_df['Organism'] = neuro_rec_df['Organism'].replace({
        'C. Elegans': 'C. elegans',
        'Zebrafish Larvae': 'Zebrafish',
    })
    neuro_rec_df = neuro_rec_df.rename(columns={'Fixated / moving': 'FixMov'})

    sim_data_columns = [
        'Connectivity accuracy', 'Percentage of neurons', 'Neurontypes',
        'Plasticity', 'Functional Accuracy', 'Neuromodulation',
        'Temporal resolution', 'Behavior', 'Personality-defining Characteristics', 'Learning',
    ]
    sim_column_labels = [textwrap.fill(col, 15, break_long_words=False) for col in sim_data_columns]

    rec_data_columns = ['Temporal resolution: Hz', 'Duration single session min per Individual',
                        'Resolution in µm isotropic', 'Number of neurons']

    # Convert to numeric
    for col in sim_data_columns:
        neuro_sim_df[col] = pd.to_numeric(neuro_sim_df[col], errors='coerce')
    for col in rec_data_columns:
        if col in neuro_rec_df.columns:
            neuro_rec_df[col] = pd.to_numeric(neuro_rec_df[col], errors='coerce')

    fig, axs = plt.subplots(5, 3, figsize=(16, 24), subplot_kw=dict(polar=True))

    for i, organism in enumerate(organisms):
        # Simulations column
        organism_sim_df = neuro_sim_df[neuro_sim_df['Organism_mapped'] == organism]
        organism_sim_df = organism_sim_df.dropna(subset=sim_data_columns)

        N_sim = len(sim_data_columns)
        angles_sim = np.linspace(0, 2 * np.pi, N_sim, endpoint=False)
        width_sim = 2 * np.pi / N_sim

        ax = axs[i, 0]
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, [3]*100, color=COLORS['border'], lw=2)

        for j in range(N_sim):
            theta_div = 2*np.pi * (j-1/2) / N_sim
            ax.plot([theta_div, theta_div], [0, 3], color=COLORS['border'], lw=1)

        if not organism_sim_df.empty:
            num_sims = len(organism_sim_df)
            normalized_alpha = 1 - (1 - 0.7)**(1/max(num_sims, 1))
            for _, row in organism_sim_df.iterrows():
                try:
                    values = row[sim_data_columns].astype(float).values
                    ax.bar(x=angles_sim, height=values, width=width_sim, bottom=0.0,
                           alpha=normalized_alpha, color=TEAL, edgecolor='none')
                except:
                    pass

        ax.set_xticks(angles_sim)
        ax.set_xticklabels(sim_column_labels, fontsize=8, color=COLORS['text'])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_ylim(0, 3)

        # Recording columns (fixated and moving)
        organism_rec_df = neuro_rec_df[neuro_rec_df['Organism'] == organism]

        for j, fixmov in enumerate(['fixated', 'moving']):
            ax = axs[i, j+1]
            sub_df = organism_rec_df[organism_rec_df['FixMov'] == fixmov]
            sub_df = sub_df.dropna(subset=rec_data_columns)

            N_rec = len(rec_data_columns)
            angles_rec = np.linspace(0, 2 * np.pi, N_rec, endpoint=False)
            width_rec = 2 * np.pi / N_rec

            ax.plot(theta, [5]*100, color=COLORS['border'], lw=2)
            for k in range(N_rec):
                theta_div = 2*np.pi * (k-1/2) / N_rec
                ax.plot([theta_div, theta_div], [0, 5], color=COLORS['border'], lw=1)

            if not sub_df.empty:
                num_recs = len(sub_df)
                normalized_alpha = 1 - (1 - 0.7)**(1/max(num_recs, 1))
                for _, row in sub_df.iterrows():
                    values = []
                    for col in rec_data_columns:
                        val = row[col]
                        if pd.notna(val) and val > 0:
                            log_val = np.log10(val)
                            normalized = min(5, max(0, (log_val + 2) * 1.0))
                            values.append(normalized)
                        else:
                            values.append(0)
                    ax.bar(x=angles_rec, height=values, width=width_rec, bottom=0.0,
                           alpha=normalized_alpha, color=GOLD, edgecolor='none')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 5)

    # Add row labels
    for i, label in enumerate(organisms):
        fig.text(-0.02, 1 - (i+0.5)/5, label, rotation=90, va='center', fontweight='bold', fontsize=14)

    # Add column labels
    col_labels = ['Simulations', 'Recordings (fixated)', 'Recordings (moving)']
    for j, label in enumerate(col_labels):
        fig.text(0.17 + j/3, 1.01, label, ha='center', va='top', fontweight='bold', fontsize=14)

    plt.tight_layout()
    add_attribution(fig)
    fig.savefig('../output/all-sim-rec.svg', format='svg', bbox_inches='tight', pad_inches=0.1)
    fig.savefig('../output/all-sim-rec.png', format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("   Done!")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 16: Funding figure
# =============================================================================
print("\n[16/16] Generating: funding.svg/png")
try:
    # Load funding data from new data structure
    neuro_proj_df = pd.read_csv(
        '../data/State of Brain Emulation Report 2025 Data Repository - Costs Neuroscience Megaprojects.csv'
    )
    other_proj_df = pd.read_csv(
        '../data/State of Brain Emulation Report 2025 Data Repository - Costs Non-Neuroscience Megaprojects.csv'
    )

    # Clean neuroscience projects data
    neuro_proj_df = neuro_proj_df.rename(columns={
        'project name': 'Name',
        'Neuroscience funding ($M)': 'Budget_M',
        'start year': 'StartYear',
        'end year': 'EndYear',
    })
    neuro_proj_df['Category'] = 'Neuroscience'
    neuro_proj_df['Budget_M'] = pd.to_numeric(neuro_proj_df['Budget_M'], errors='coerce')
    neuro_proj_df['StartYear'] = pd.to_numeric(neuro_proj_df['StartYear'], errors='coerce')

    # Clean other projects data
    other_proj_df = other_proj_df.rename(columns={
        'project': 'Name',
        'cost ($B, 2024 dollars)': 'Budget_B',
        'field': 'Category',
    })
    other_proj_df['Budget_M'] = pd.to_numeric(other_proj_df['Budget_B'], errors='coerce') * 1000

    # Combine datasets (using available columns)
    neuro_clean = neuro_proj_df[['Name', 'Budget_M', 'Category']].dropna(subset=['Budget_M'])
    other_clean = other_proj_df[['Name', 'Budget_M', 'Category']].dropna(subset=['Budget_M'])

    all_funding = pd.concat([neuro_clean, other_clean], ignore_index=True)

    if len(all_funding) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by budget for better visualization
        all_funding_sorted = all_funding.sort_values('Budget_M', ascending=True).tail(20)

        colors = [GOLD if cat == 'Neuroscience' else COLORS['caption']
                  for cat in all_funding_sorted['Category']]

        bars = ax.barh(all_funding_sorted['Name'], all_funding_sorted['Budget_M'], color=colors, alpha=0.8)

        ax.set_xscale('log')
        ax.set_xlabel('Budget (Million $)')
        ax.set_title('Megaproject Budgets Comparison')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=GOLD, label='Neuroscience'),
            Patch(facecolor=COLORS['caption'], label='Other')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True)

        plt.tight_layout()
        save_figure(fig, 'funding')
        plt.close()
        print("   Done!")
    else:
        print("   Skipped - no valid data")
except Exception as e:
    print(f"   Error: {e}")

# =============================================================================
# Figure 17: Organism Compute Requirements
# =============================================================================
print("\n[17/17] Generating: organism-compute.svg/png")
try:
    # Load computational demands data
    compute_df = pd.read_csv(
        '../data/State of Brain Emulation Report 2025 Data Repository - Computational Demands Organisms.csv'
    )

    # Parse the data - it's in a wide format with organisms as columns
    organisms_row = compute_df.iloc[0]  # neurons row
    synapses_row = compute_df.iloc[1]   # synapses row

    # Extract organism data (skip the first column which is label)
    organism_names = ['C. elegans', 'Fly', 'Mouse (cortex)', 'Mouse (brain)', 'Human (cortex)', 'Human (brain)']
    organism_cols = ['C. elegans (body)', 'fly (brain)', 'mouse (cortex)', 'mouse (brain)', 'human (cortex)', 'human (brain)']

    neurons = []
    synapses = []
    for col in organism_cols:
        if col in compute_df.columns:
            neurons.append(pd.to_numeric(organisms_row[col], errors='coerce'))
            synapses.append(pd.to_numeric(synapses_row[col], errors='coerce'))

    if len(neurons) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(organism_names))
        width = 0.6

        # Plot neurons
        axes[0].bar(x, neurons, width, color=TEAL, alpha=0.8, edgecolor='white')
        axes[0].set_yscale('log')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(organism_names, rotation=30, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Neuron Count by Organism')

        # Plot synapses
        axes[1].bar(x, synapses, width, color=GOLD, alpha=0.8, edgecolor='white')
        axes[1].set_yscale('log')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(organism_names, rotation=30, ha='right')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Synapse Count by Organism')

        plt.tight_layout()
        save_figure(fig, 'organism-compute')
        plt.close()
        print("   Done!")
    else:
        print("   Skipped - no valid data")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Figure generation complete!")
print("=" * 60)
print("\nOutput files saved to: ../output/")

# =============================================================================
# Build Download ZIPs
# =============================================================================
print("\n" + "=" * 60)
print("Building downloadable ZIP files...")
print("=" * 60)

try:
    from build_downloads import main as build_downloads
    build_downloads()
except Exception as e:
    print(f"Error building downloads: {e}")
    print("You can run 'python build_downloads.py' separately to generate ZIPs.")
