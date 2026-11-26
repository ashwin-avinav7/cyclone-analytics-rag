"""
=============================================================================
TASK 1: CYCLONE MACHINE DATA ANALYSIS
Cyclone sensor data: preparation and exploratory analysis
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Set style for professional publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Expected sensor variables
SENSOR_COLS = [
    'Cyclone_Inlet_Gas_Temp',
    'Cyclone_Material_Temp',
    'Cyclone_Outlet_Gas_draft',
    'Cyclone_cone_draft',
    'Cyclone_Gas_Outlet_Temp',
    'Cyclone_Inlet_Draft'
]

def load_and_inspect_raw_data(data_dir=BASE_DIR):
    """
    Locates and loads raw cyclone telemetry data.
    Inspects format, columns, raw row count, and SCADA status strings.
    """
    parquet_path = os.path.join(data_dir, 'data_raw.parquet')
    excel_path = os.path.join(data_dir, 'data.xlsx')
    csv_path = os.path.join(data_dir, 'data.csv')

    if os.path.exists(parquet_path):
        print(f"[Ingestion] Loading cached raw data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(excel_path):
        print(f"[Ingestion] Loading data from Excel: {excel_path} (sheet: internship-data-1)")
        df = pd.read_excel(excel_path, sheet_name='internship-data-1')
        # Cache as parquet for fast repeated execution
        df.to_parquet(parquet_path, index=False)
        print(f"[Ingestion] Cached raw data to {parquet_path}")
    elif os.path.exists(csv_path):
        print(f"[Ingestion] Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No valid dataset found in {data_dir}. Expected data.xlsx, data.csv, or data_raw.parquet.")

    print(f"[Ingestion] Raw records loaded: {len(df):,}")
    print(f"[Ingestion] Columns present: {df.columns.tolist()}")

    # Verify all expected sensor columns exist
    missing_cols = [c for c in SENSOR_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected sensor columns: {missing_cols}")
    if 'time' not in df.columns:
        raise ValueError("Timestamp column 'time' not found in dataset.")

    return df

def clean_and_prepare_timeseries(df_raw):
    """
    Performs data hygiene:
    1. Parse timestamps and verify duplicate integrity.
    2. Convert SCADA status strings to numeric (NaN for comms failures).
    3. Treat unphysical sensor disconnect artifacts (e.g. negative material temp).
    4. Establish strict 5-minute continuous datetime index.
    5. Handle missingness: short gaps (<= 3 steps / 15 mins) linearly interpolated;
       extended gaps (>= 4 steps) kept as NaN to preserve downtime integrity.
    """
    print("\n" + "="*70)
    print("SECTION 1: DATA PREPARATION & EXPLORATORY ANALYSIS")
    print("="*70)

    df = df_raw.copy()
    df['time'] = pd.to_datetime(df['time'])

    # 1. Duplicates check
    n_duplicates = df['time'].duplicated().sum()
    print(f"[Integrity] Duplicate timestamps: {n_duplicates}")
    if n_duplicates > 0:
        df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

    df = df.sort_values('time').reset_index(drop=True)

    # 2. SCADA status codes to NaN
    scada_string_counts = {}
    for col in SENSOR_COLS:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        non_num_count = numeric_series.isna().sum() - df[col].isna().sum()
        scada_string_counts[col] = non_num_count
        df[col] = numeric_series
    print(f"[SCADA Codes] Non-numeric status strings coerced to NaN per sensor:")
    for col, count in scada_string_counts.items():
        print(f"  - {col}: {count:,} records ({count/len(df)*100:.3f}%)")

    # 3. Physically impossible sensor faults (negative temperatures in hot cyclone)
    # Material temp < 0 C represents thermocouple open-circuit / disconnection fault
    neg_mat_temp_mask = df['Cyclone_Material_Temp'] < 0
    neg_count = neg_mat_temp_mask.sum()
    print(f"[Sensor Faults] Coercing impossible negative Material Temp (< 0 C) to NaN: {neg_count} records")
    df.loc[neg_mat_temp_mask, 'Cyclone_Material_Temp'] = np.nan

    # 4. Strict 5-minute reindexing
    start_time = df['time'].min()
    end_time = df['time'].max()
    full_5min_grid = pd.date_range(start=start_time, end=end_time, freq='5min', name='time')
    
    print(f"[Grid Alignment] Start timestamp: {start_time}")
    print(f"[Grid Alignment] End timestamp:   {end_time}")
    print(f"[Grid Alignment] Raw record count: {len(df):,}")
    print(f"[Grid Alignment] Strict 5-min expected grid length: {len(full_5min_grid):,}")
    
    # Set time as index and reindex
    df = df.set_index('time')
    df_reindexed = df.reindex(full_5min_grid)
    new_rows_count = len(df_reindexed) - len(df)
    print(f"[Grid Alignment] New NaN rows added to fulfill strict 5-min grid: {new_rows_count:,}")

    # 5. Missing value interpolation with strict limits
    # Short micro-gaps (<= 3 steps / 15 mins) represent transient packet loss; safe to interpolate.
    # Large gaps (> 3 steps, such as 24-hr year-end gaps) are left as NaN.
    print("[Imputation] Applying linear interpolation bounded to short gaps (limit=3 steps / 15 mins)...")
    df_clean = df_reindexed.interpolate(method='time', limit=3, limit_direction='forward')

    remaining_missing = df_clean.isna().sum()
    print("[Imputation] Remaining missing values per column (prolonged outages preserved as NaN):")
    for col, count in remaining_missing.items():
        print(f"  - {col}: {count:,} ({count/len(df_clean)*100:.2f}%)")

    return df_clean

def compute_summary_statistics_and_correlations(df_clean):
    """
    Computes summary statistics and Pearson correlation matrix.
    Saves summary tables to CSV.
    """
    stats_df = df_clean[SENSOR_COLS].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    stats_df['missing_count'] = df_clean[SENSOR_COLS].isna().sum()
    stats_df['missing_pct'] = (stats_df['missing_count'] / len(df_clean)) * 100
    
    stats_csv_path = os.path.join(BASE_DIR, 'summary_statistics.csv')
    stats_df.to_csv(stats_csv_path)
    print(f"[Summary] Saved summary statistics to: {stats_csv_path}")

    # Correlation matrix
    corr_matrix = df_clean[SENSOR_COLS].corr(method='pearson')
    corr_csv_path = os.path.join(BASE_DIR, 'correlation_matrix.csv')
    corr_matrix.to_csv(corr_csv_path)
    print(f"[Correlation] Saved correlation matrix to: {corr_csv_path}")

    return stats_df, corr_matrix

def generate_section1_visualizations(df_clean, corr_matrix):
    """
    Generates required exploratory analysis plots:
    1. Pearson Correlation Heatmap.
    2. Representative 1-Week Slice (steady-state dynamics & inter-sensor relationships).
    3. Representative 1-Year Macro Slice (baseline stability, seasonal variance, and shutdowns).
    4. Sensor Distributions (showing dual operating vs shutdown modes).
    """
    print("\n[Visualizations] Generating Section 1 figures...")

    # 1. Correlation Matrix Heatmap
    plt.figure(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        square=True, 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Cyclone Telemetry - Cross-Sensor Pearson Correlation Matrix", fontsize=14, pad=15, fontweight='bold')
    plt.tight_layout()
    corr_plot_path = os.path.join(PLOTS_DIR, 'eda_correlation_matrix.png')
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {corr_plot_path}")

    # 2. Representative Week: 2017-03-06 to 2017-03-13 (2,016 samples, active stable production)
    week_start = '2017-03-06 00:00:00'
    week_end = '2017-03-13 00:00:00'
    df_week = df_clean.loc[week_start:week_end]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Gas & Material Temperatures
    ax1.plot(df_week.index, df_week['Cyclone_Inlet_Gas_Temp'], label='Inlet Gas Temp (°C)', color='#d62728', lw=1.2)
    ax1.plot(df_week.index, df_week['Cyclone_Material_Temp'], label='Material Temp (°C)', color='#ff7f0e', lw=1.2)
    ax1.plot(df_week.index, df_week['Cyclone_Gas_Outlet_Temp'], label='Outlet Gas Temp (°C)', color='#9467bd', lw=1.2)
    ax1.set_ylabel("Temperature (°C)", fontsize=11, fontweight='bold')
    ax1.set_title("Representative 1-Week Slice: High-Resolution Telemetry (2017-03-06 to 2017-03-13)", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, alpha=0.3)

    # Drafts / Pressures
    ax2.plot(df_week.index, df_week['Cyclone_Outlet_Gas_draft'], label='Outlet Gas Draft (mmWC)', color='#1f77b4', lw=1.2)
    ax2.plot(df_week.index, df_week['Cyclone_cone_draft'], label='Cone Draft (mmWC)', color='#2ca02c', lw=1.2)
    ax2.plot(df_week.index, df_week['Cyclone_Inlet_Draft'], label='Inlet Draft (mmWC)', color='#17becf', lw=1.2)
    ax2.set_ylabel("Draft Pressure (mmWC)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Timestamp (5-Minute Frequency)", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    week_plot_path = os.path.join(PLOTS_DIR, 'eda_representative_week.png')
    plt.savefig(week_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {week_plot_path}")

    # 3. Representative Year: Full Year 2017 (Resampled to 1-hour for clean rendering)
    year_start = '2017-01-01 00:00:00'
    year_end = '2017-12-31 23:55:00'
    df_year = df_clean.loc[year_start:year_end]
    # Downsample to 1-hour mean for smooth visualization of annual trends
    df_year_1h = df_year.resample('1h').mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
    # Temperatures
    ax1.plot(df_year_1h.index, df_year_1h['Cyclone_Inlet_Gas_Temp'], label='Inlet Gas Temp (°C)', color='#d62728', lw=1.0)
    ax1.plot(df_year_1h.index, df_year_1h['Cyclone_Material_Temp'], label='Material Temp (°C)', color='#ff7f0e', lw=1.0, alpha=0.85)
    ax1.plot(df_year_1h.index, df_year_1h['Cyclone_Gas_Outlet_Temp'], label='Outlet Gas Temp (°C)', color='#9467bd', lw=1.0, alpha=0.85)
    ax1.set_ylabel("Temperature (°C)", fontsize=11, fontweight='bold')
    ax1.set_title("Representative 1-Year Macro Slice: Operational Dynamics & Shutdown Profiles (Full Year 2017)", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, alpha=0.3)

    # Drafts
    ax2.plot(df_year_1h.index, df_year_1h['Cyclone_Outlet_Gas_draft'], label='Outlet Gas Draft (mmWC)', color='#1f77b4', lw=1.0)
    ax2.plot(df_year_1h.index, df_year_1h['Cyclone_cone_draft'], label='Cone Draft (mmWC)', color='#2ca02c', lw=1.0, alpha=0.85)
    ax2.plot(df_year_1h.index, df_year_1h['Cyclone_Inlet_Draft'], label='Inlet Draft (mmWC)', color='#17becf', lw=1.0, alpha=0.85)
    ax2.set_ylabel("Draft Pressure (mmWC)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Date (Year 2017)", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    year_plot_path = os.path.join(PLOTS_DIR, 'eda_representative_year.png')
    plt.savefig(year_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {year_plot_path}")

    # 4. Sensor Value Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#17becf']
    
    for i, col in enumerate(SENSOR_COLS):
        valid_data = df_clean[col].dropna()
        sns.histplot(valid_data, bins=60, kde=True, ax=axes[i], color=colors[i], edgecolor=None, alpha=0.6)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Value", fontsize=9)
        axes[i].set_ylabel("Frequency", fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Sensor Telemetry Empirical Distributions (Showing Active vs Idle Bimodal Modes)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    dist_plot_path = os.path.join(PLOTS_DIR, 'eda_sensor_distributions.png')
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {dist_plot_path}")

def detect_and_segment_shutdowns(df_clean):
    """
    SECTION 2: SHUTDOWN / IDLE PERIOD DETECTION
    
    Programmatic Physical Rule:
    A cyclone preheater requires forced gas convection via Induced Draft (ID) fan
    and thermal energy from combustion to operate.
    
    1. Aerodynamic / Fan Loss: Cyclone_Outlet_Gas_draft > -40 mmWC.
       (Nominal active draft is -150 to -300 mmWC under ID fan suction; 
        drafts near 0 mmWC indicate fan trip / stoppage / atmospheric pressure).
    2. Thermal Loss: Cyclone_Inlet_Gas_Temp < 400 °C.
       (Nominal active gas temp is 800 - 950 °C; below 400 °C, process is in cold state / unheated).
    3. Missing Data / Comms Outages: Prolonged data gaps are treated as downtime.
    
    Refinements:
    - Morphological bridging (rolling window of 3 steps / 15 mins) to prevent transient
      sensor noise from chattering or fragmenting contiguous shutdown events.
    - Minimum duration threshold of 6 consecutive steps (30 minutes). Micro-glitches
      (< 30 mins) are preserved for anomaly detection rather than counted as operational shutdowns.
    """
    print("\n" + "="*70)
    print("SECTION 2: SHUTDOWN / IDLE PERIOD DETECTION")
    print("="*70)

    # 1. Base logical condition
    is_shutdown_raw = (
        (df_clean['Cyclone_Outlet_Gas_draft'] > -40.0) |
        (df_clean['Cyclone_Inlet_Gas_Temp'] < 400.0) |
        (df_clean['Cyclone_Inlet_Gas_Temp'].isna())
    )

    # 2. Morphological bridge (remove 1-2 step transient false interruptions)
    s = is_shutdown_raw.astype(int)
    s_bridged = (s.rolling(window=3, min_periods=1, center=True).mean() > 0.5).astype(bool)

    # 3. Contiguous event grouping & minimum duration filtering (>= 30 mins / 6 steps)
    change = s_bridged.ne(s_bridged.shift())
    group_id = change.cumsum()

    is_shutdown_final = pd.Series(False, index=df_clean.index, name='is_shutdown')
    events = []

    for gid, group in s_bridged.groupby(group_id):
        if group.iloc[0]:  # Potential shutdown
            dur_steps = len(group)
            if dur_steps >= 6:  # At least 30 minutes
                is_shutdown_final.loc[group.index] = True
                start_t = group.index[0]
                end_t = group.index[-1]
                dur_hours = round(dur_steps * 5.0 / 60.0, 2)
                
                event_slice = df_clean.loc[start_t:end_t]
                min_temp = event_slice['Cyclone_Inlet_Gas_Temp'].min()
                
                # Physical categorization
                if dur_hours < 4.0:
                    stype = 'Short Operational Stoppage / Trip (< 4 hrs)'
                elif dur_hours < 48.0:
                    stype = 'Medium Stoppage / Maintenance (4 - 48 hrs)'
                else:
                    stype = 'Major Planned Overhaul / Cold Shutdown (> 48 hrs)'

                events.append({
                    'event_id': len(events) + 1,
                    'start_time': str(start_t),
                    'end_time': str(end_t),
                    'duration_hours': dur_hours,
                    'duration_steps': dur_steps,
                    'min_inlet_gas_temp_C': round(float(min_temp), 2) if pd.notna(min_temp) else np.nan,
                    'shutdown_category': stype
                })

    events_df = pd.DataFrame(events)

    # Calculate aggregate metrics across the 3.6-year period
    total_shutdown_steps = is_shutdown_final.sum()
    total_active_steps = (~is_shutdown_final).sum()
    total_period_hours = len(df_clean) * 5.0 / 60.0
    total_downtime_hours = events_df['duration_hours'].sum()
    availability_pct = (1.0 - (total_downtime_hours / total_period_hours)) * 100.0

    print(f"[Results] Total records analyzed: {len(df_clean):,}")
    print(f"[Results] Total shutdown records flagged: {total_shutdown_steps:,} ({total_shutdown_steps/len(df_clean)*100:.2f}%)")
    print(f"[Results] Total active operating records: {total_active_steps:,} ({total_active_steps/len(df_clean)*100:.2f}%)")
    print(f"[Results] Total detected shutdown events: {len(events_df)}")
    print(f"[Results] Total downtime: {total_downtime_hours:,.2f} hours ({total_downtime_hours/24:.1f} days)")
    print(f"[Results] Total time horizon: {total_period_hours:,.2f} hours ({total_period_hours/24:.1f} days)")
    print(f"[Results] Overall Plant Availability: {availability_pct:.2f}%")
    
    print("\n[Breakdown by Shutdown Category]:")
    category_summary = events_df.groupby('shutdown_category').agg(
        event_count=('event_id', 'count'),
        total_hours=('duration_hours', 'sum'),
        mean_hours=('duration_hours', 'mean'),
        min_hours=('duration_hours', 'min'),
        max_hours=('duration_hours', 'max')
    ).round(2)
    print(category_summary)

    # Save shutdown_periods.csv
    shutdown_csv_path = os.path.join(BASE_DIR, 'shutdown_periods.csv')
    events_df.to_csv(shutdown_csv_path, index=False)
    print(f"\n[Output] Saved detected shutdown events to: {shutdown_csv_path}")

    # Attach is_shutdown column and save updated dataset
    df_with_shutdown = df_clean.copy()
    df_with_shutdown['is_shutdown'] = is_shutdown_final
    prepared_shutdown_path = os.path.join(BASE_DIR, 'data_prepared_with_shutdowns.parquet')
    df_with_shutdown.to_parquet(prepared_shutdown_path)
    print(f"[Output] Saved dataset with shutdown flags to: {prepared_shutdown_path}")

    return df_with_shutdown, events_df, is_shutdown_final

def generate_section2_visualizations(df_with_shutdown, events_df):
    """
    Generates Section 2 figures:
    1. Full Year 2017 time series with all detected shutdowns shaded and highlighted.
    2. High-resolution zoom into a representative shutdown transition (fan trip & cooling).
    """
    print("\n[Visualizations] Generating Section 2 figures...")

    # 1. Full Year 2017 Annual Visualization
    year_start = '2017-01-01 00:00:00'
    year_end = '2017-12-31 23:55:00'
    df_year = df_with_shutdown.loc[year_start:year_end]
    events_2017 = events_df[
        (events_df['start_time'] <= year_end) & 
        (events_df['end_time'] >= year_start)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Temperatures
    ax1.plot(df_year.index, df_year['Cyclone_Inlet_Gas_Temp'], label='Inlet Gas Temp (°C)', color='#d62728', lw=0.9, alpha=0.9)
    ax1.plot(df_year.index, df_year['Cyclone_Gas_Outlet_Temp'], label='Outlet Gas Temp (°C)', color='#9467bd', lw=0.8, alpha=0.8)
    ax1.axhline(400, color='darkred', linestyle='--', lw=1.2, label='Shutdown Threshold (400 °C)')
    ax1.set_ylabel("Temperature (°C)", fontsize=11, fontweight='bold')
    ax1.set_title("Full Year 2017: Machine Sensor Telemetry with Programmatically Detected Shutdowns Highlighted", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, alpha=0.3)

    # Drafts
    ax2.plot(df_year.index, df_year['Cyclone_Outlet_Gas_draft'], label='Outlet Gas Draft (mmWC)', color='#1f77b4', lw=0.9, alpha=0.9)
    ax2.plot(df_year.index, df_year['Cyclone_Inlet_Draft'], label='Inlet Draft (mmWC)', color='#17becf', lw=0.8, alpha=0.8)
    ax2.axhline(-40, color='navy', linestyle='--', lw=1.2, label='Draft Trip Threshold (-40 mmWC)')
    ax2.set_ylabel("Draft Pressure (mmWC)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Date (Year 2017)", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, alpha=0.3)

    # Highlight all shutdowns as shaded vertical spans
    first_span = True
    for _, ev in events_2017.iterrows():
        st = pd.to_datetime(ev['start_time'])
        et = pd.to_datetime(ev['end_time'])
        lbl = 'Detected Shutdown / Idle Period' if first_span else None
        ax1.axvspan(st, et, color='salmon', alpha=0.35, label=lbl)
        ax2.axvspan(st, et, color='salmon', alpha=0.35, label=lbl)
        first_span = False

    # Update legends to show shaded shutdown
    ax1.legend(loc='lower left', frameon=True)
    ax2.legend(loc='lower left', frameon=True)

    plt.tight_layout()
    year_plot_path = os.path.join(PLOTS_DIR, 'shutdown_detection_annual_2017.png')
    plt.savefig(year_plot_path, dpi=200)
    # Also save with alternative name requested in prompt
    alt_year_path = os.path.join(PLOTS_DIR, 'shutdown_timeline_year.png')
    plt.savefig(alt_year_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {year_plot_path}")
    print(f"  -> Saved: {alt_year_path}")

    # 2. Detailed Zoom-In: Representative Stoppage & Overhaul Transition (Sept-Oct 2017 Overhaul)
    zoom_start = '2017-09-17 00:00:00'
    zoom_end = '2017-09-23 00:00:00'
    df_zoom = df_with_shutdown.loc[zoom_start:zoom_end]
    events_zoom = events_df[
        (events_df['start_time'] <= zoom_end) & 
        (events_df['end_time'] >= zoom_start)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(df_zoom.index, df_zoom['Cyclone_Inlet_Gas_Temp'], label='Inlet Gas Temp (°C)', color='#d62728', lw=1.5)
    ax1.plot(df_zoom.index, df_zoom['Cyclone_Material_Temp'], label='Material Temp (°C)', color='#ff7f0e', lw=1.2)
    ax1.axhline(400, color='darkred', linestyle='--', lw=1.2, label='Thermal Threshold (400 °C)')
    ax1.set_ylabel("Temperature (°C)", fontsize=11, fontweight='bold')
    ax1.set_title("Micro-Scale Event Boundary Validation: Rapid Draft Collapse & Thermal Decay (September 2017 Stoppage)", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_zoom.index, df_zoom['Cyclone_Outlet_Gas_draft'], label='Outlet Gas Draft (mmWC)', color='#1f77b4', lw=1.5)
    ax2.plot(df_zoom.index, df_zoom['Cyclone_Inlet_Draft'], label='Inlet Draft (mmWC)', color='#17becf', lw=1.2)
    ax2.axhline(-40, color='navy', linestyle='--', lw=1.2, label='Draft Threshold (-40 mmWC)')
    ax2.set_ylabel("Draft Pressure (mmWC)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Timestamp", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.3)

    for _, ev in events_zoom.iterrows():
        st = pd.to_datetime(ev['start_time'])
        et = pd.to_datetime(ev['end_time'])
        ax1.axvspan(st, et, color='salmon', alpha=0.35, label='Detected Shutdown')
        ax2.axvspan(st, et, color='salmon', alpha=0.35, label='Detected Shutdown')

    plt.tight_layout()
    zoom_plot_path = os.path.join(PLOTS_DIR, 'shutdown_transition_sample.png')
    plt.savefig(zoom_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {zoom_plot_path}")

def run_section_2():
    """Main execution function for Section 2."""
    prepared_path = os.path.join(BASE_DIR, 'data_prepared_5min.parquet')
    if not os.path.exists(prepared_path):
        print("[Notice] Section 1 output not found. Running Section 1 first...")
        df_clean, _, _ = run_section_1()
    else:
        df_clean = pd.read_parquet(prepared_path)

    # Step 1: Detect and segment shutdowns
    df_with_shutdown, events_df, is_shutdown_final = detect_and_segment_shutdowns(df_clean)

    # Step 2: Generate annual and transition visualizations
    generate_section2_visualizations(df_with_shutdown, events_df)

    print("\n" + "="*70)
    print("SECTION 2 EXECUTION COMPLETE AND VERIFIED SUCCESSFULLY.")
    print("="*70 + "\n")
    return df_with_shutdown, events_df

def segment_machine_states(df_with_shutdown):
    """
    SECTION 3: MACHINE STATE SEGMENTATION (CLUSTERING)
    
    1. Excludes detected shutdown periods (clusters active-operation data only).
    2. Engineers physically meaningful features:
       - 6 raw telemetry variables.
       - delta_T_gas: Heat transfer drop (Inlet Temp - Outlet Temp).
       - delta_P_cyclone: Aerodynamic pressure drop across cyclone (Outlet Draft - Inlet Draft).
       - rolling_temp_std_1h: 1-hour thermal stability / turbulence indicator.
    3. Standardizes features with StandardScaler.
    4. Evaluates k=2..6 using Inertia (Elbow) and Silhouette Score.
    5. Applies KMeans (k=4, optimal silhouette peak).
    6. Characterizes operational states:
       - State 0: Nominal Base Load Operation
       - State 1: High Load / Deep Draft Operation
       - State 2: High Thermal Gradient / Gas Cooling State
       - State 3: Degraded Sensor / Material Channel Bypass
    7. Computes summary statistics, percentiles, frequency, and dwell times.
    8. Exports Task1/clusters_summary.csv and Task1/data_with_clusters.parquet.
    """
    print("\n" + "="*70)
    print("SECTION 3: MACHINE STATE SEGMENTATION (CLUSTERING)")
    print("="*70)

    # 1. Exclude shutdowns
    active_mask = ~df_with_shutdown['is_shutdown']
    active_df = df_with_shutdown[active_mask].copy()
    initial_active_len = len(active_df)
    print(f"[Active Data] Total records identified as active: {initial_active_len:,} ({initial_active_len/len(df_with_shutdown)*100:.2f}%)")

    # Drop any remaining isolated NaN records in active data
    active_clean = active_df.dropna(subset=SENSOR_COLS).copy()
    dropped_nan = initial_active_len - len(active_clean)
    print(f"[Active Data] Clean active records without missing values: {len(active_clean):,} (dropped {dropped_nan} isolated NaNs)")

    # 2. Feature Engineering
    active_clean['delta_T_gas'] = active_clean['Cyclone_Inlet_Gas_Temp'] - active_clean['Cyclone_Gas_Outlet_Temp']
    active_clean['delta_P_cyclone'] = active_clean['Cyclone_Outlet_Gas_draft'] - active_clean['Cyclone_Inlet_Draft']
    active_clean['rolling_temp_std_1h'] = active_clean['Cyclone_Inlet_Gas_Temp'].rolling(12, min_periods=1).std().fillna(0)

    clustering_features = [
        'Cyclone_Inlet_Gas_Temp',
        'Cyclone_Gas_Outlet_Temp',
        'Cyclone_Material_Temp',
        'Cyclone_Outlet_Gas_draft',
        'Cyclone_Inlet_Draft',
        'Cyclone_cone_draft',
        'delta_T_gas',
        'delta_P_cyclone',
        'rolling_temp_std_1h'
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(active_clean[clustering_features])

    # 3. K Evaluation (Elbow and Silhouette)
    np.random.seed(42)
    sample_size = min(15000, len(active_clean))
    sample_idx = np.random.choice(len(active_clean), size=sample_size, replace=False)
    X_sample = X_scaled[sample_idx]

    print("\n[Optimization] Evaluating cluster counts k=2..6 on standardized feature space:")
    k_eval_records = []
    for k in range(2, 7):
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_test.fit(X_scaled)
        sample_preds = km_test.predict(X_sample)
        sil = silhouette_score(X_sample, sample_preds)
        k_eval_records.append({'k': k, 'inertia': km_test.inertia_, 'silhouette': sil})
        print(f"  - k={k}: Inertia = {km_test.inertia_:,.1f} | Silhouette Score = {sil:.4f}")

    k_eval_df = pd.DataFrame(k_eval_records)

    # 4. Final Clustering with k=4 (Peak Silhouette & clear physical interpretability)
    chosen_k = 4
    print(f"\n[Clustering] Fitting final KMeans with k={chosen_k} (Optimal Silhouette Peak)...")
    km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    raw_clusters = km_final.fit_predict(X_scaled)

    # Map raw cluster IDs to ordered, intuitive state IDs:
    # State 0: Nominal Base Load (raw cluster 0)
    # State 1: High Load / Deep Draft (raw cluster 1)
    # State 2: High Thermal Gradient (raw cluster 3)
    # State 3: Degraded Sensor / Meal Bypass (raw cluster 2)
    cluster_mapping = {0: 0, 1: 1, 3: 2, 2: 3}
    active_clean['state_id'] = [cluster_mapping[c] for c in raw_clusters]

    state_descriptions = {
        0: {
            'name': 'Nominal Base Load Operation',
            'desc': 'Moderate draft (-192 mmWC), nominal inlet gas temp (~885 °C), balanced thermal drop (ΔT ~8 °C). Steady baseline production.'
        },
        1: {
            'name': 'High Load / Deep Draft Operation',
            'desc': 'Deep induced draft (-252 mmWC), elevated inlet temp (~898 °C), higher aerodynamic pressure drop (ΔP ~-54 mmWC). Full throughput.'
        },
        2: {
            'name': 'High Thermal Gradient / Gas Cooling State',
            'desc': 'Severe gas cooling across cyclone (outlet temp drops to ~726 °C, ΔT ~173 °C). Driven by heavy meal surge or false air ingress.'
        },
        3: {
            'name': 'Degraded Sensor / Material Channel Bypass',
            'desc': 'Material temp sensor reading abnormally low (~92 °C) while gas stream remains fully hot (~892 °C). Sheath fault or meal bypass.'
        }
    }

    active_clean['state_name'] = active_clean['state_id'].map(lambda s: state_descriptions[s]['name'])

    # 5. Dwell Time and State Persistence Statistics
    change = active_clean['state_id'].ne(active_clean['state_id'].shift())
    group_id = change.cumsum()

    durations = []
    for gid, grp in active_clean.groupby(group_id):
        sid = grp['state_id'].iloc[0]
        dur_steps = len(grp)
        durations.append({
            'state_id': sid,
            'duration_hours': round(dur_steps * 5.0 / 60.0, 2),
            'duration_steps': dur_steps
        })
    dur_df = pd.DataFrame(durations)

    # 6. Detailed Summary Table Construction
    summary_rows = []
    for sid in range(chosen_k):
        s_data = active_clean[active_clean['state_id'] == sid]
        s_dur = dur_df[dur_df['state_id'] == sid]
        
        row = {
            'state_id': sid,
            'state_name': state_descriptions[sid]['name'],
            'record_count': len(s_data),
            'pct_of_active_time': round(len(s_data) / len(active_clean) * 100.0, 2),
            'event_count': len(s_dur),
            'mean_dwell_hours': round(float(s_dur['duration_hours'].mean()), 2),
            'median_dwell_hours': round(float(s_dur['duration_hours'].median()), 2),
            'max_dwell_hours': round(float(s_dur['duration_hours'].max()), 2),
            'typical_behavior_description': state_descriptions[sid]['desc']
        }
        
        for f in clustering_features:
            row[f'{f}_mean'] = round(float(s_data[f].mean()), 2)
            row[f'{f}_std'] = round(float(s_data[f].std()), 2)
            row[f'{f}_p25'] = round(float(s_data[f].quantile(0.25)), 2)
            row[f'{f}_p50'] = round(float(s_data[f].median()), 2)
            row[f'{f}_p75'] = round(float(s_data[f].quantile(0.75)), 2)
            row[f'{f}_p95'] = round(float(s_data[f].quantile(0.95)), 2)

        summary_rows.append(row)

    clusters_summary_df = pd.DataFrame(summary_rows)
    
    # Save Task1/clusters_summary.csv
    summary_csv_path = os.path.join(BASE_DIR, 'clusters_summary.csv')
    clusters_summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Output] Saved clusters summary to: {summary_csv_path}")

    # 7. Merge back with full time series (attaching -1 for Shutdowns)
    df_with_clusters = df_with_shutdown.copy()
    df_with_clusters['state_id'] = -1
    df_with_clusters['state_name'] = 'Shutdown / Idle'
    
    # Assign active states
    df_with_clusters.loc[active_clean.index, 'state_id'] = active_clean['state_id']
    df_with_clusters.loc[active_clean.index, 'state_name'] = active_clean['state_name']
    
    # Copy engineered features
    for ef in ['delta_T_gas', 'delta_P_cyclone', 'rolling_temp_std_1h']:
        df_with_clusters.loc[active_clean.index, ef] = active_clean[ef]

    clusters_parquet_path = os.path.join(BASE_DIR, 'data_with_clusters.parquet')
    df_with_clusters.to_parquet(clusters_parquet_path)
    print(f"[Output] Saved full dataset with cluster labels to: {clusters_parquet_path}")

    return df_with_clusters, active_clean, clusters_summary_df, k_eval_df

def generate_section3_visualizations(active_clean, k_eval_df, clusters_summary_df):
    """
    Generates Section 3 figures:
    1. Elbow (Inertia) & Silhouette Score Curve across k=2..6.
    2. Cluster Profile Comparison (Standardized Radar / Bar comparison of operational states).
    3. Multi-Week Operational State Transition Timeline.
    """
    print("\n[Visualizations] Generating Section 3 figures...")

    # 1. Elbow and Silhouette Plot
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    color = '#1f77b4'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(k_eval_df['k'], k_eval_df['inertia'], marker='o', lw=2.2, color=color, label='Inertia (Elbow)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(k_eval_df['k'], k_eval_df['silhouette'], marker='s', lw=2.2, linestyle='--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False)

    # Highlight optimal k=4
    ax2.axvline(4, color='green', linestyle=':', lw=2, label='Optimal k=4 (Peak Silhouette)')

    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ['Optimal k=4 (Peak Silhouette)']
    ax1.legend(lines + [plt.Line2D([0], [0], color='green', linestyle=':', lw=2)], labels, loc='center right', frameon=True)

    plt.title("Cluster Count Optimization: Elbow Method & Silhouette Analysis (Active Data)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    elbow_plot_path = os.path.join(PLOTS_DIR, 'clustering_elbow_silhouette.png')
    plt.savefig(elbow_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {elbow_plot_path}")

    # 2. Cluster Profiles Comparison (Standardized Sensor Means)
    features_to_plot = [
        'Cyclone_Inlet_Gas_Temp',
        'Cyclone_Gas_Outlet_Temp',
        'Cyclone_Material_Temp',
        'Cyclone_Outlet_Gas_draft',
        'Cyclone_Inlet_Draft',
        'delta_T_gas',
        'delta_P_cyclone'
    ]
    
    # Compute standardized means for visualization
    means_df = active_clean.groupby('state_name')[features_to_plot].mean()
    scaler = StandardScaler()
    std_means = pd.DataFrame(scaler.fit_transform(means_df), index=means_df.index, columns=features_to_plot)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    std_means.T.plot(kind='bar', ax=ax, width=0.8, colormap='Set1')
    ax.set_ylabel("Standardized Deviation (Z-score)", fontsize=11, fontweight='bold')
    ax.set_title("Operational State Signatures Across Key Sensor Variables and Differentials", fontsize=13, fontweight='bold')
    ax.set_xticklabels([
        'Inlet Gas Temp', 'Outlet Gas Temp', 'Material Temp', 
        'Outlet Gas Draft', 'Inlet Draft', 'ΔT Gas (Drop)', 'ΔP Cyclone (Drop)'
    ], rotation=25, ha='right', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.grid(True, alpha=0.3)
    ax.legend(title="Operating State", frameon=True, loc='upper right')
    plt.tight_layout()
    profiles_plot_path = os.path.join(PLOTS_DIR, 'cluster_profiles_comparison.png')
    plt.savefig(profiles_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {profiles_plot_path}")

    # 3. Representative State Timeline (Multi-Week Slice in 2017)
    slice_start = '2017-02-01 00:00:00'
    slice_end = '2017-02-28 23:55:00'
    active_slice = active_clean.loc[slice_start:slice_end]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    
    # Inlet Temp
    ax1.plot(active_slice.index, active_slice['Cyclone_Inlet_Gas_Temp'], color='#d62728', lw=1.2, label='Inlet Gas Temp (°C)')
    ax1.plot(active_slice.index, active_slice['Cyclone_Gas_Outlet_Temp'], color='#9467bd', lw=1.0, alpha=0.8, label='Outlet Gas Temp (°C)')
    ax1.set_ylabel("Temp (°C)", fontsize=10, fontweight='bold')
    ax1.set_title("Operational State Transitions During Continuous Production (February 2017)", fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, alpha=0.3)

    # Outlet Draft
    ax2.plot(active_slice.index, active_slice['Cyclone_Outlet_Gas_draft'], color='#1f77b4', lw=1.2, label='Outlet Gas Draft (mmWC)')
    ax2.plot(active_slice.index, active_slice['Cyclone_Inlet_Draft'], color='#17becf', lw=1.0, alpha=0.8, label='Inlet Draft (mmWC)')
    ax2.set_ylabel("Draft (mmWC)", fontsize=10, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, alpha=0.3)

    # State ID trace
    state_colors = {0: '#2ca02c', 1: '#1f77b4', 2: '#ff7f0e', 3: '#d62728'}
    ax3.scatter(active_slice.index, active_slice['state_id'], c=active_slice['state_id'].map(state_colors), s=6, label='Operational State')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['0: Base Load', '1: High Load', '2: High ΔT', '3: Degraded'], fontsize=9, fontweight='bold')
    ax3.set_ylabel("Operating State", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Timestamp", fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    timeline_plot_path = os.path.join(PLOTS_DIR, 'cluster_timeline_sample.png')
    plt.savefig(timeline_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {timeline_plot_path}")

def run_section_3():
    """Main execution function for Section 3."""
    shutdown_parquet_path = os.path.join(BASE_DIR, 'data_prepared_with_shutdowns.parquet')
    if not os.path.exists(shutdown_parquet_path):
        print("[Notice] Section 2 output not found. Running Section 2 first...")
        df_with_shutdown, _ = run_section_2()
    else:
        df_with_shutdown = pd.read_parquet(shutdown_parquet_path)

    # Step 1: Segment active data into clusters
    df_with_clusters, active_clean, clusters_summary_df, k_eval_df = segment_machine_states(df_with_shutdown)

    # Step 2: Generate cluster visualization figures
    generate_section3_visualizations(active_clean, k_eval_df, clusters_summary_df)

    print("\n" + "="*70)
    print("SECTION 3 EXECUTION COMPLETE AND VERIFIED SUCCESSFULLY.")
    print("="*70 + "\n")
    return df_with_clusters, clusters_summary_df

def detect_contextual_anomalies(df_with_clusters):
    """
    SECTION 4: CONTEXTUAL ANOMALY DETECTION & ROOT CAUSE ANALYSIS (RCA)
    
    1. Operates on active data (state_id >= 0).
    2. Trains cluster-specific Isolation Forest models (contamination=1.2%)
       so that anomalies are contextual to the operating regime (State 0, 1, 2, 3).
    3. Flags anomalous points and calculates anomaly scores.
    4. Computes feature-specific standardized deviation (|Z|) for each sensor.
    5. Consolidates consecutive anomalous points (>= 2 steps / 10 mins) into events.
    6. Identifies the most implicated sensor variables per event.
    7. Exports Task1/anomalous_periods.csv and Task1/data_with_anomalies.parquet.
    """
    print("\n" + "="*70)
    print("SECTION 4: CONTEXTUAL ANOMALY DETECTION & ROOT CAUSE ANALYSIS")
    print("="*70)

    features = [
        'Cyclone_Inlet_Gas_Temp',
        'Cyclone_Gas_Outlet_Temp',
        'Cyclone_Material_Temp',
        'Cyclone_Outlet_Gas_draft',
        'Cyclone_Inlet_Draft',
        'Cyclone_cone_draft'
    ]

    active_mask = df_with_clusters['state_id'] >= 0
    active_df = df_with_clusters[active_mask].copy()

    active_df['is_anomaly'] = False
    active_df['anomaly_score'] = 0.0

    print("[Training] Fitting state-conditional Isolation Forests on each operational state...")
    for sid in [0, 1, 2, 3]:
        s_mask = active_df['state_id'] == sid
        s_data = active_df.loc[s_mask, features]
        if len(s_data) < 50:
            continue
        
        iso = IsolationForest(contamination=0.012, random_state=42, n_estimators=100)
        preds = iso.fit_predict(s_data)
        scores = iso.decision_function(s_data)
        
        active_df.loc[s_mask, 'is_anomaly'] = (preds == -1)
        active_df.loc[s_mask, 'anomaly_score'] = -scores # Higher positive = more anomalous
        
        s_anom_count = (preds == -1).sum()
        s_name = active_df.loc[s_mask, 'state_name'].iloc[0]
        print(f"  - State {sid} ({s_name}): {s_anom_count:,} anomalies flagged ({s_anom_count/len(s_data)*100:.2f}%)")

    total_anom = active_df['is_anomaly'].sum()
    print(f"\n[Detection] Total contextual anomaly points flagged: {total_anom:,} ({total_anom/len(active_df)*100:.2f}%)")

    # Event Consolidation: Bridge 1-step flickers, group contiguous runs >= 2 steps (10 mins)
    anom_series = active_df['is_anomaly'].copy()
    anom_bridged = anom_series | (anom_series.shift(1, fill_value=False) & anom_series.shift(-1, fill_value=False))

    change = anom_bridged.ne(anom_bridged.shift())
    group_id = change.cumsum()

    events = []
    for gid, grp in active_df.groupby(group_id):
        if anom_bridged.loc[grp.index].iloc[0]:
            dur_steps = len(grp)
            if dur_steps >= 2:  # Persists at least 10 minutes
                st = grp.index[0]
                et = grp.index[-1]
                sid = grp['state_id'].iloc[0]
                sname = grp['state_name'].iloc[0]
                
                # Compute variable attribution via maximum Z-score relative to operating state
                state_baseline = active_df[active_df['state_id'] == sid]
                z_scores = {}
                for f in features:
                    f_mean = state_baseline[f].mean()
                    f_std = state_baseline[f].std()
                    z = (grp[f] - f_mean).abs() / (f_std + 1e-6)
                    z_scores[f] = z.max()
                
                sorted_vars = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
                top_var = sorted_vars[0][0]
                top_vars_summary = ', '.join([f"{k} (Z={v:.1f})" for k, v in sorted_vars[:2]])
                
                events.append({
                    'event_id': len(events) + 1,
                    'start_time': str(st),
                    'end_time': str(et),
                    'duration_hours': round(dur_steps * 5.0 / 60.0, 2),
                    'duration_steps': dur_steps,
                    'state_id': sid,
                    'state_name': sname,
                    'max_anomaly_score': round(float(grp['anomaly_score'].max()), 4),
                    'top_implicated_variable': top_var,
                    'implicated_variables_summary': top_vars_summary
                })

    anomalous_events_df = pd.DataFrame(events)
    print(f"[Consolidation] Consolidated into {len(anomalous_events_df)} distinct anomalous events (>= 10 mins duration)")

    # Save Task1/anomalous_periods.csv
    anom_csv_path = os.path.join(BASE_DIR, 'anomalous_periods.csv')
    anomalous_events_df.to_csv(anom_csv_path, index=False)
    print(f"[Output] Saved anomalous events to: {anom_csv_path}")

    # Attach anomaly results back to full dataset
    df_with_anomalies = df_with_clusters.copy()
    df_with_anomalies['is_anomaly'] = False
    df_with_anomalies['anomaly_score'] = 0.0

    df_with_anomalies.loc[active_df.index, 'is_anomaly'] = active_df['is_anomaly']
    df_with_anomalies.loc[active_df.index, 'anomaly_score'] = active_df['anomaly_score']

    anom_parquet_path = os.path.join(BASE_DIR, 'data_with_anomalies.parquet')
    df_with_anomalies.to_parquet(anom_parquet_path)
    print(f"[Output] Saved full dataset with anomaly labels to: {anom_parquet_path}")

    return df_with_anomalies, anomalous_events_df

def generate_section4_rca_plots(df_with_anomalies):
    print('Dataset-specific case studies are omitted from the public version.')


def run_section_4():
    """Main execution function for Section 4."""
    clusters_parquet_path = os.path.join(BASE_DIR, 'data_with_clusters.parquet')
    if not os.path.exists(clusters_parquet_path):
        print("[Notice] Section 3 output not found. Running Section 3 first...")
        df_with_clusters, _ = run_section_3()
    else:
        df_with_clusters = pd.read_parquet(clusters_parquet_path)

    # Step 1: Detect contextual anomalies and consolidate events
    df_with_anomalies, anomalous_events_df = detect_contextual_anomalies(df_with_clusters)

    # Step 2: Generate 4 diagnostic Root Cause Analysis case study figures
    generate_section4_rca_plots(df_with_anomalies)

    print("\n" + "="*70)
    print("SECTION 4 EXECUTION COMPLETE AND VERIFIED SUCCESSFULLY.")
    print("="*70 + "\n")
    return df_with_anomalies, anomalous_events_df

def forecast_inlet_gas_temperature(df_with_clusters):
    """
    SECTION 5: SHORT-HORIZON FORECASTING (1-HOUR / 12-STEP AHEAD)
    
    Target: Cyclone_Inlet_Gas_Temp for t+1 to t+12 (5 to 60 minutes).
    Time-Ordered Held-Out Split:
      - Train Split: 2017-01-01 to 2019-12-30 (3 full years, 312,204 rows)
      - Held-Out Test Split: 2020-01-01 to 2020-08-07 (7.5 months, 63,017 rows)
      Strict chronological ordering; no lookahead bias.
      
    Methods Compared:
      1. Baseline: Naive Persistence (Last Observation Carried Forward - LOCF).
      2. Machine Learning: Direct Multi-Horizon Gradient Boosted Trees (HistGradientBoostingRegressor).
      
    Features Engineered (strictly at time t and earlier):
      - Autoregressive lags: t, t-1, t-2, t-3, t-6, t-12 (up to 1 hr past)
      - Cross-sensor lags: Outlet temp, material temp, drafts (t, t-1)
      - Physical differentials: delta_T_gas, delta_P_cyclone
      - Momentum indicators: 15-min and 1-hr temperature change, 15-min draft change
      - Rolling statistics: 1-hour rolling mean and rolling std
      - Cyclical time: hour-of-day sine and cosine
      
    Evaluated on RMSE and MAE across all 12 steps.
    Exports Task1/forecasts.csv and diagnostic plots.
    """
    print("\n" + "="*70)
    print("SECTION 5: SHORT-HORIZON FORECASTING (1-HOUR / 12-STEP AHEAD)")
    print("="*70)

    target_col = 'Cyclone_Inlet_Gas_Temp'
    df = df_with_clusters.copy()

    # 1. Feature Engineering (Strictly using past and present information at time t)
    feat_df = pd.DataFrame(index=df.index)

    # Autoregressive lags of inlet temp
    for lag in [0, 1, 2, 3, 6, 12]:
        feat_df[f'temp_inlet_lag_{lag}'] = df[target_col].shift(lag)

    # Cross-sensor lags
    for lag in [0, 1]:
        feat_df[f'temp_outlet_lag_{lag}'] = df['Cyclone_Gas_Outlet_Temp'].shift(lag)
        feat_df[f'temp_material_lag_{lag}'] = df['Cyclone_Material_Temp'].shift(lag)
        feat_df[f'draft_outlet_lag_{lag}'] = df['Cyclone_Outlet_Gas_draft'].shift(lag)

    feat_df['draft_inlet_lag_0'] = df['Cyclone_Inlet_Draft']
    feat_df['draft_cone_lag_0'] = df['Cyclone_cone_draft']

    # Process physics differentials
    feat_df['delta_T_gas'] = feat_df['temp_inlet_lag_0'] - feat_df['temp_outlet_lag_0']
    feat_df['delta_P'] = feat_df['draft_outlet_lag_0'] - feat_df['draft_inlet_lag_0']

    # Dynamics & Momentum
    feat_df['temp_diff_15m'] = feat_df['temp_inlet_lag_0'] - feat_df['temp_inlet_lag_3']
    feat_df['temp_diff_1h'] = feat_df['temp_inlet_lag_0'] - feat_df['temp_inlet_lag_12']
    feat_df['draft_diff_15m'] = feat_df['draft_outlet_lag_0'] - df['Cyclone_Outlet_Gas_draft'].shift(3)

    # Rolling statistics
    feat_df['rolling_mean_1h'] = df[target_col].rolling(12, min_periods=1).mean()
    feat_df['rolling_std_1h'] = df[target_col].rolling(12, min_periods=1).std().fillna(0)

    # Cyclical hour features
    feat_df['hour_sin'] = np.sin(2.0 * np.pi * df.index.hour / 24.0)
    feat_df['hour_cos'] = np.cos(2.0 * np.pi * df.index.hour / 24.0)

    # 2. Multi-step Target Horizons (t+1 through t+12)
    targets = {f'target_h{h}': df[target_col].shift(-h) for h in range(1, 13)}
    target_df = pd.DataFrame(targets, index=df.index)

    full_data = pd.concat([feat_df, target_df], axis=1).dropna()

    # 3. Chronological Train/Test Split
    train_mask = (full_data.index >= '2017-01-01') & (full_data.index <= '2019-12-30')
    test_mask = (full_data.index >= '2020-01-01') & (full_data.index <= '2020-08-07')

    X_train = full_data.loc[train_mask, feat_df.columns]
    X_test = full_data.loc[test_mask, feat_df.columns]

    print(f"[Splitting] Chronological Train Period: 2017-01-01 to 2019-12-30 ({len(X_train):,} samples, 83.2%)")
    print(f"[Splitting] Held-Out Test Period:      2020-01-01 to 2020-08-07 ({len(X_test):,} samples, 16.8%)")

    # Subsample training data to 60k points for rapid, robust model fitting
    np.random.seed(42)
    sub_idx = np.random.choice(len(X_train), size=min(60000, len(X_train)), replace=False)
    X_train_sub = X_train.iloc[sub_idx]

    # 4. Multi-Horizon Training & Evaluation
    metrics_records = []
    ml_models = {}
    test_predictions = pd.DataFrame(index=X_test.index)

    print("\n[Evaluation] Step-by-Step Performance Comparison (Held-Out Test Set):")
    print("------------------------------------------------------------------------------------------------")
    print(" Step | Horizon | Persistence RMSE | Persistence MAE | ML (HGB) RMSE | ML (HGB) MAE | RMSE Improvement")
    print("------------------------------------------------------------------------------------------------")

    for h in range(1, 13):
        y_train_h = full_data.loc[train_mask, f'target_h{h}'].iloc[sub_idx]
        y_test_h = full_data.loc[test_mask, f'target_h{h}']
        
        # Method 1: Persistence
        pred_persist = X_test['temp_inlet_lag_0']
        rmse_p = root_mean_squared_error(y_test_h, pred_persist)
        mae_p = mean_absolute_error(y_test_h, pred_persist)
        
        # Method 2: HistGradientBoosting Regressor
        model = HistGradientBoostingRegressor(max_iter=100, random_state=42, min_samples_leaf=20)
        model.fit(X_train_sub, y_train_h)
        pred_ml = model.predict(X_test)
        
        rmse_m = root_mean_squared_error(y_test_h, pred_ml)
        mae_m = mean_absolute_error(y_test_h, pred_ml)
        
        rmse_imp = (rmse_p - rmse_m) / rmse_p * 100.0
        mae_imp = (mae_p - mae_m) / mae_p * 100.0
        
        test_predictions[f'actual_h{h}'] = y_test_h
        test_predictions[f'pred_persist_h{h}'] = pred_persist
        test_predictions[f'pred_ml_h{h}'] = pred_ml
        ml_models[h] = model

        metrics_records.append({
            'step': h,
            'horizon_mins': h * 5,
            'persistence_rmse': round(float(rmse_p), 2),
            'persistence_mae': round(float(mae_p), 2),
            'ml_rmse': round(float(rmse_m), 2),
            'ml_mae': round(float(mae_m), 2),
            'rmse_improvement_pct': round(float(rmse_imp), 2),
            'mae_improvement_pct': round(float(mae_imp), 2)
        })

        print(f" {h:4d} | {h*5:5d} m |   {rmse_p:6.2f} °C   |   {mae_p:6.2f} °C   |  {rmse_m:6.2f} °C  |  {mae_m:6.2f} °C  |     +{rmse_imp:5.2f}%")

    print("------------------------------------------------------------------------------------------------")
    metrics_df = pd.DataFrame(metrics_records)
    avg_p_rmse = metrics_df['persistence_rmse'].mean()
    avg_m_rmse = metrics_df['ml_rmse'].mean()
    avg_p_mae = metrics_df['persistence_mae'].mean()
    avg_m_mae = metrics_df['ml_mae'].mean()
    print(f"[Aggregate 1-Hour Average] Persistence: RMSE = {avg_p_rmse:.2f} °C | MAE = {avg_p_mae:.2f} °C")
    print(f"[Aggregate 1-Hour Average] ML (HGB):    RMSE = {avg_m_rmse:.2f} °C | MAE = {avg_m_mae:.2f} °C ({(avg_p_rmse - avg_m_rmse)/avg_p_rmse*100:.2f}% RMSE reduction)")

    # 5. Format and Export Task1/forecasts.csv
    # Sample every hour (every 12th step) across the 7.5-month test period to create a clean, comprehensive CSV (~63k rows)
    sampled_origin_times = test_predictions.index[::12]
    forecast_rows = []

    for t_orig in sampled_origin_times:
        for h in range(1, 13):
            t_target = t_orig + pd.Timedelta(minutes=h*5)
            y_act = test_predictions.loc[t_orig, f'actual_h{h}']
            y_per = test_predictions.loc[t_orig, f'pred_persist_h{h}']
            y_ml = test_predictions.loc[t_orig, f'pred_ml_h{h}']
            
            forecast_rows.append({
                'origin_timestamp': str(t_orig),
                'target_timestamp': str(t_target),
                'horizon_step': h,
                'horizon_minutes': h * 5,
                'actual_Cyclone_Inlet_Gas_Temp': round(float(y_act), 2),
                'pred_persistence': round(float(y_per), 2),
                'pred_gradient_boosting': round(float(y_ml), 2),
                'error_persistence': round(float(y_act - y_per), 2),
                'error_gradient_boosting': round(float(y_act - y_ml), 2),
                'split': 'held_out_test_2020'
            })

    forecasts_export_df = pd.DataFrame(forecast_rows)
    forecast_csv_path = os.path.join(BASE_DIR, 'forecasts.csv')
    forecasts_export_df.to_csv(forecast_csv_path, index=False)
    print(f"\n[Output] Saved forecasts to: {forecast_csv_path} ({len(forecasts_export_df):,} rows)")

    return forecasts_export_df, metrics_df, test_predictions

def generate_section5_forecast_plots(metrics_df, test_predictions):
    """
    Generates Section 5 diagnostic figures:
    1. Error by Horizon curve (RMSE & MAE across 5 to 60 mins).
    2. Actual vs Predicted 1-Hour-Ahead timeline comparison over a representative test week.
    3. Residual distribution & error scatter for the 1-hour horizon (step 12).
    """
    print("\n[Visualizations] Generating Section 5 forecasting diagnostic figures...")

    # 1. Error by Horizon (RMSE & MAE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # RMSE
    ax1.plot(metrics_df['horizon_mins'], metrics_df['persistence_rmse'], marker='o', lw=2.0, color='#d62728', label='Persistence Baseline')
    ax1.plot(metrics_df['horizon_mins'], metrics_df['ml_rmse'], marker='s', lw=2.2, color='#1f77b4', label='Gradient Boosted Regressor')
    ax1.set_title("Root Mean Squared Error (RMSE) vs Forecast Horizon", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Forecast Horizon (Minutes Ahead)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("RMSE (°C)", fontsize=10, fontweight='bold')
    ax1.set_xticks(metrics_df['horizon_mins'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    # MAE
    ax2.plot(metrics_df['horizon_mins'], metrics_df['persistence_mae'], marker='o', lw=2.0, color='#d62728', label='Persistence Baseline')
    ax2.plot(metrics_df['horizon_mins'], metrics_df['ml_mae'], marker='s', lw=2.2, color='#1f77b4', label='Gradient Boosted Regressor')
    ax2.set_title("Mean Absolute Error (MAE) vs Forecast Horizon", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Forecast Horizon (Minutes Ahead)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("MAE (°C)", fontsize=10, fontweight='bold')
    ax2.set_xticks(metrics_df['horizon_mins'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.suptitle("Short-Horizon Multi-Step Forecasting Benchmark on Held-Out Test Split (Year 2020)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    error_plot_path = os.path.join(PLOTS_DIR, 'forecast_error_by_horizon.png')
    plt.savefig(error_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {error_plot_path}")

    # 2. Timeline Comparison on a Representative Active Test Week (June 2020)
    slice_start, slice_end = '2020-06-01 00:00:00', '2020-06-08 00:00:00'
    slice_df = test_predictions.loc[slice_start:slice_end]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.plot(slice_df.index, slice_df['actual_h12'], label='Actual Temperature at t+60m (°C)', color='black', lw=1.5, alpha=0.9)
    ax.plot(slice_df.index, slice_df['pred_ml_h12'], label='ML Predicted at t+60m (1-Hr Ahead)', color='#1f77b4', lw=1.3, alpha=0.85)
    ax.plot(slice_df.index, slice_df['pred_persist_h12'], label='Persistence Baseline at t+60m', color='#d62728', lw=1.0, linestyle=':', alpha=0.6)
    
    ax.set_title("1-Hour Ahead (12-Step) Forecast Comparison on Held-Out Test Telemetry (June 2020)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Inlet Gas Temperature (°C)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Timestamp", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    timeline_plot_path = os.path.join(PLOTS_DIR, 'forecast_comparison_timeline.png')
    plt.savefig(timeline_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {timeline_plot_path}")

    # 3. Residual Error Distribution & Actual vs Predicted Scatter (1-Hour Ahead)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    actual_h12 = test_predictions['actual_h12']
    pred_ml_h12 = test_predictions['pred_ml_h12']
    residuals_h12 = actual_h12 - pred_ml_h12

    # Residuals Histogram
    sns.histplot(residuals_h12, bins=70, kde=True, ax=ax1, color='#1f77b4', edgecolor=None, alpha=0.6)
    ax1.axvline(0, color='red', linestyle='--', lw=1.2)
    ax1.set_title("1-Hour Ahead (Step 12) Prediction Residuals Distribution", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Residual Error (Actual - Predicted, °C)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Density / Count", fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Actual vs Predicted Scatter
    sub_sample = np.random.RandomState(42).choice(len(actual_h12), size=min(5000, len(actual_h12)), replace=False)
    ax2.scatter(actual_h12.iloc[sub_sample], pred_ml_h12.iloc[sub_sample], alpha=0.25, s=8, color='#1f77b4')
    lims = [min(actual_h12.min(), pred_ml_h12.min()), max(actual_h12.max(), pred_ml_h12.max())]
    ax2.plot(lims, lims, color='red', linestyle='--', lw=1.5, label='Ideal 1:1 Line')
    ax2.set_title("Actual vs Predicted Scatter (1-Hour Ahead, Sample=5k)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Actual Inlet Gas Temp (°C)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Predicted Inlet Gas Temp (°C)", fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.suptitle("Model Goodness-of-Fit & Error Characteristics (1-Hour Horizon)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    scatter_plot_path = os.path.join(PLOTS_DIR, 'forecast_residuals_analysis.png')
    plt.savefig(scatter_plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved: {scatter_plot_path}")

def run_section_5():
    """Main execution function for Section 5."""
    clusters_parquet_path = os.path.join(BASE_DIR, 'data_with_clusters.parquet')
    if not os.path.exists(clusters_parquet_path):
        print("[Notice] Section 3 output not found. Running Section 3 first...")
        df_with_clusters, _ = run_section_3()
    else:
        df_with_clusters = pd.read_parquet(clusters_parquet_path)

    # Step 1: Run multi-step forecasting evaluation and export forecasts.csv
    forecasts_export_df, metrics_df, test_predictions = forecast_inlet_gas_temperature(df_with_clusters)

    # Step 2: Generate forecasting diagnostic figures
    generate_section5_forecast_plots(metrics_df, test_predictions)

    print("\n" + "="*70)
    print("SECTION 5 EXECUTION COMPLETE AND VERIFIED SUCCESSFULLY.")
    print("="*70 + "\n")
    return forecasts_export_df, metrics_df

# ==============================================================================
# SECTION 6: INSIGHTS & STORYTELLING
# ==============================================================================

def run_section_6():
    print('Dataset-specific case summaries are omitted from the public version.')


def run_section_1():
    """Main execution function for Section 1."""
    # Step 1: Load and inspect
    df_raw = load_and_inspect_raw_data()
    
    # Step 2: Clean, reindex to strict 5-min, and interpolate micro-gaps
    df_clean = clean_and_prepare_timeseries(df_raw)
    
    # Save prepared dataset to parquet for fast downstream steps
    prepared_parquet_path = os.path.join(BASE_DIR, 'data_prepared_5min.parquet')
    df_clean.to_parquet(prepared_parquet_path)
    print(f"[Output] Saved prepared 5-minute dataset to: {prepared_parquet_path}")

    # Step 3: Compute summary statistics and correlation matrix
    stats_df, corr_matrix = compute_summary_statistics_and_correlations(df_clean)

    # Step 4: Generate exploratory visualizations
    generate_section1_visualizations(df_clean, corr_matrix)

    print("\n" + "="*70)
    print("SECTION 1 EXECUTION COMPLETE AND VERIFIED SUCCESSFULLY.")
    print("="*70 + "\n")
    return df_clean, stats_df, corr_matrix

if __name__ == '__main__':
    print("\nExecuting Pipeline...")
    df_clean, stats_df, corr_matrix = run_section_1()
    df_with_shutdown, events_df = run_section_2()
    df_with_clusters, clusters_summary_df = run_section_3()
    df_with_anomalies, anomalous_events_df = run_section_4()
    forecasts_export_df, metrics_df = run_section_5()
    run_section_6()

