# MSE 433 Module 4 -- Outlier Analysis of PFA Ablation Procedures

**Course:** MSE 433 -- Applications of Management Engineering, University of Waterloo
**Instructor:** Prof. Houra Mahmoudzadeh
**Case Credit:** Dr. William K. Chan, MD FRCPC -- Cardiologist and Electrophysiologist, Waterloo Regional Health Network

## Background

Atrial fibrillation (AFib) is the most common sustained cardiac arrhythmia, affecting over 46 million people worldwide. This case study examines **Pulse-Field Ablation (PFA)** procedures performed with the **Varipulse** catheter device at the **Waterloo Regional Health Network Electrophysiology (EP) Lab**. The dataset contains 150 consecutive PFA cases with granular timing breakdowns for every procedural phase and the total patient in-to-out duration (`PT IN-OUT`).

The goal is to identify which cases are statistical outliers in procedure duration, understand *why* they are outliers using machine-learning explainability, and surface actionable operational insights for the EP Lab.

---

## Data Definitions

Each row in the dataset is one PFA procedure. Definitions from `Data/MSE433_M4_Definitions.pdf`:

| Column | Definition |
|---|---|
| **PT PREP/INTUBATION** | Pt-In to Access. Patient positioning, monitoring hookup, anesthesia induction, intubation, sterile draping. |
| **ACCESS** | Femoral vein puncture and sheath insertion (vascular access). |
| **TSP** | Transseptal puncture -- crossing from right to left atrium. |
| **PRE-MAP** | Electroanatomic mapping of the left atrium before ablation begins. |
| **ABL DURATION** | Abl Start to End. Total elapsed time from first to last ablation delivery, including catheter repositioning. |
| **ABL TIME** | Cumulative active "pulse on" energy delivery time. Shorter than ABL DURATION because it excludes repositioning. |
| **#ABL** | Number of individual ablation sites targeted. |
| **#APPLICATIONS** | Total PFA pulse train applications delivered (always 3x #ABL per protocol). |
| **LA DWELL TIME** | Abl Start to Cath-Out. Total time catheter is inside the left atrium during the therapeutic phase. *(Excluded from model -- aggregate of ABL DURATION + post-ablation catheter work.)* |
| **CASE TIME** | Cath In to Cath Out. Core procedural duration. *(Excluded from model -- aggregate sub-total, corr 0.91 with target.)* |
| **SKIN-SKIN** | Access to Cath-Out. Similar to CASE TIME but measured from skin puncture. *(Excluded from model -- aggregate sub-total, corr 0.92 with target.)* |
| **POST CARE/EXTUBATION** | Cath-Out to Pt-Out. Hemostasis, extubation, brief post-procedure monitoring. |
| **PT IN-OUT** | **Target variable.** Total patient time in the lab from arrival to departure (sum of all phases). |
| **Note** | Flags for additional ablation targets beyond standard PVI: CTI, BOX, PST BOX, SVC, AAFL, or TROUBLESHOOT. Blank = standard PVI only. |

### Procedure Timeline

```
PT IN-OUT (target) = PT PREP/INTUBATION + ACCESS + [CASE TIME] + POST CARE/EXTUBATION
                                                        |
                            CASE TIME (excluded) = TSP + PRE-MAP + LA DWELL TIME
                                                                      |
                             LA DWELL TIME (excluded) = ABL DURATION + post-abl catheter work
```

**Why CASE TIME, SKIN-SKIN, and LA DWELL TIME are excluded from the model:** These are aggregate sub-totals that directly sum into PT IN-OUT. SHAP flagging them would only say "the case was long because the case was long." By excluding them, SHAP points to the specific granular phases (prep, access, TSP, mapping, ablation, post-care) where time is actually being spent.

---

## Process and Methodology

### 1. Data Loading and Cleaning

- Loaded 150 PFA cases from `Data/MSE433_M4_Data.xlsx`.
- Parsed multi-row headers, converted all timing columns to numeric.
- Removed running-average columns (`AVG CASE TIME`, `AVG SKIN-SKIN`, `AVG TURNOVER TIME`) and mostly-null columns (`PT OUT TIME`).
- Dropped rows missing the target variable, retaining **145 usable cases**.

### 2. Exploratory Data Analysis

- **Distribution analysis:** Histogram, boxplot, and Q-Q plot of `PT IN-OUT` revealed a right-skewed distribution (skewness = 2.45, kurtosis = 9.76) with a heavy upper tail.
- **Outlier method comparison:** Evaluated three approaches side-by-side:
  - *IQR (1.5x):* 5 outliers (3.4%)
  - *Z-score (|Z| > 2):* 5 outliers (3.4%)
  - *90th percentile:* 15 outliers (10.3%)
- **Correlation heatmap** of all timing features to identify collinearity and redundant aggregates.
- **Per-physician distribution comparisons** (boxplots and histograms).

### 3. Outlier Classification

Two complementary outlier definitions are used:

| Scope | Method | Threshold | Outliers |
|---|---|---|---|
| **Global** | 90th percentile | >= 99 min (90th percentile = 98.6 min) | 15 cases (10.3%) |
| **Dr. A** | IQR (Q3 + 1.0 x IQR) | > 89 min | 4 cases |
| **Dr. B** | IQR (Q3 + 1.0 x IQR) | > 120 min | 6 cases |
| **Dr. C** | IQR (Q3 + 1.0 x IQR) | > 102 min | 1 case |

- **Global (90th percentile):** Captures enough positive cases (15) for stable LightGBM modeling. Binary target: Normal vs Outlier.
- **Per-physician (IQR):** Q3 + 1.0 x IQR applied within each physician's own distribution. Thresholds scale with individual caseload and variability rather than being dominated by the global pool. Per-physician outliers use Q3 + 1.0×IQR (tighter than the standard 1.5× to capture enough outliers for modeling with small per-physician samples).

### 4. Feature Engineering

14 features in total:

| Category | Features | Count |
|---|---|---:|
| Granular procedural timings | PT PREP/INTUBATION, ACCESS, TSP, PRE-MAP, ABL DURATION, ABL TIME, #ABL, POST CARE/EXTUBATION | 8 |
| Physician encoding | Physician identifier (label-encoded) | 1 |
| Note-derived binary flags | CTI, BOX, PST, SVC (additional procedures performed) | 4 |
| Scheduling | Case order within the day (1st, 2nd, 3rd, ...) | 1 |

**Excluded:** CASE TIME, SKIN-SKIN, LA DWELL TIME (aggregate sub-totals that leak the target). #APPLICATIONS excluded — deterministic function of #ABL (always 3x per protocol, r=1.0 correlation).

### 5. LightGBM Classification

- Gradient-boosted decision tree classifier with balanced class weights.
- Models fitted on the **full dataset** (no train/test split or cross-validation) because the goal is SHAP interpretation of what drives outlier status, not out-of-sample prediction.
- Per-physician models fitted for any physician with at least 2 outlier cases.

### 6. SHAP Explainability

- `TreeExplainer` applied to each fitted LightGBM model.
- Outputs per model:
  - **Beeswarm (summary) plot** -- feature value vs. SHAP impact for every case.
  - **Bar plot** -- mean absolute SHAP values (global feature importance).
  - **Dependence plots** -- top 4 features with interaction coloring.

### 7. Additional Analyses

- **Learning curve:** Linear trendline and 10-case rolling average of `PT IN-OUT` over case sequence, per physician.
- **Case complexity:** Outlier rates broken down by additional procedure flags (CTI, BOX, PST BOX, SVC).
- **Day-of scheduling:** Mean duration and outlier rate by case position in the daily schedule.

---

## Key Findings

### Who has the problem

- **Dr. B accounts for 14 of 15 global outliers.** Physician identity is the #1 global SHAP predictor (1.045). Dr. A has zero cases above the global 99-min threshold. Dr. B's baseline distribution is substantially higher (Q3 = 97 min vs. Dr. A's Q3 = 76 min), which explains why most global outliers originate from Dr. B even when judged by their own IQR threshold (>120 min).

### What drives long cases -- and it differs by physician

- **Dr. B's bottleneck is ABL DURATION** (ablation start-to-end, per-physician IQR analysis, 6 outliers >120 min): outlier cases average 50.7 min vs. 25.3 min for normal cases (+101%). Crucially, ABL TIME (actual pulse-on energy delivery) is nearly identical between outlier and normal cases (7.5 vs. 7.7 min). This means Dr. B is spending twice as long **repositioning the catheter between ablation sites**, not delivering more energy.
- **Dr. A's bottleneck is completely different -- PT PREP/INTUBATION** (patient positioning, monitoring hookup, anesthesia induction, intubation, sterile draping). A different phase entirely. Blanket process changes would not help both physicians; interventions must be physician-specific.

### PRE-MAP is a hidden complexity signal

- PRE-MAP (pre-ablation electroanatomic mapping) shows the largest percentage difference between outlier and normal cases globally: **+272%** (5.9 vs. 1.6 min). When mapping takes longer, it signals a complex patient anatomy that cascades into longer times across every downstream phase.

### Dr. B's long durations are not explained by harder patients

- Dr. A actually performs the **most** additional procedures (20% of cases include BOX, PST BOX, or CTI targets) yet has the **shortest** average duration (69.5 min). Dr. B performs fewer additional procedures (11.7%) but takes substantially longer (91.9 min). Dr. C sees the simplest cases (0% additional procedures) with moderate duration (74.7 min).
- The difference is **catheter repositioning efficiency**: Dr. B averages 20.1 min of repositioning time vs. 13.8 min for Dr. A, despite nearly identical pulse-on time (~7 min). Dr. B is not treating harder cases -- the time is lost in the mechanics of moving the catheter between ablation sites.

### Case complexity multiplies risk

- **CTI (cavo-tricuspid isthmus) procedures carry a 60% outlier rate** vs. 7.4% for standard PVI-only cases. Adding a CTI ablation target nearly guarantees a long case. PST BOX procedures also elevate risk (14.3% outlier rate).

### Scheduling position matters

- **First case of the day averages 91 min** with a 21.2% outlier rate. Duration drops steadily to 61 min by the 7th case, with 0% outlier rate for cases 5--7. This reflects setup overhead, equipment preparation, and a warm-up effect that dissipates through the day.

### Model-based reassignment analysis

- **Reassigning Dr. B's 14 global outliers to Dr. A would resolve all 14**, as Dr. A's procedural patterns produce lower outlier risk across all timing phases.

### Everyone is improving

- All three physicians show a **negative learning curve slope** (Dr. A: --0.18 min/case, Dr. B: --0.28, Dr. C: --1.33), consistent with a procedural learning effect on the Varipulse device. Dr. C's steeper slope reflects a smaller sample (15 cases) in the early part of their learning curve.

### Actionable recommendations

1. **ABL DURATION for Dr. B:** Investigate catheter repositioning workflow. ABL TIME is constant, so the energy delivery protocol is not the issue -- it is the mechanical movement between sites.
2. **PT PREP/INTUBATION for Dr. A:** Standardize patient setup and anesthesia induction. This is Dr. A's primary bottleneck.
3. **TSP and PRE-MAP globally:** Extended transseptal puncture and mapping times are strong outlier predictors. Protocols to streamline these phases (e.g., pre-procedure imaging, standardized puncture technique) could reduce duration across all physicians.
4. **Schedule CTI cases strategically:** Given the 60% outlier rate, CTI cases should not be scheduled as the first case of the day (where setup overhead already adds time) or back-to-back with other complex cases.
5. **Reduce first-case-of-day overhead:** The 91 min average for case #1 vs. 69 min for case #6 suggests that pre-procedural setup (room prep, equipment checks, team briefing) could be front-loaded before the patient arrives.

### Notes

- **Case 38:** Case 38 was performed by Dr. D substituting for Dr. B; attributed to Dr. B in the analysis.
- **Case 57 (Dr. B, 204 min):** The longest case involved AAFL (atrial flutter ablation), an additional ablation target.
- **Case 90 (Dr. C, 103 min):** Dr. C's single outlier is flagged as TROUBLESHOOT, indicating a technical issue rather than a procedural bottleneck.

---

## How It All Fits Together

```
┌─────────────────────────┐
│   outlier_analysis.py   │  Step 1: Run the analysis pipeline
│   (Python, ~1550 lines) │  Reads: Data/MSE433_M4_Data.xlsx
└───────────┬─────────────┘  Writes: output/ (CSV, JSON, 17 PNGs)
            │
            ▼
┌─────────────────────────────────────┐
│   app/backend/                      │  Step 2: Export data for dashboard
│   ├── export_dashboard_data.py      │  Reads: output/ + Data/
│   ├── whatif_simulator.py           │  Writes: app/frontend/src/data/*.json
│   └── reassignment_data.py         │
└───────────┬─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│   app/frontend/                     │  Step 3: Interactive dashboard
│   React 19 + TypeScript + Vite      │  Imports: src/data/*.json (static)
│   Tailwind CSS + Recharts           │  Serves: localhost:5173
└─────────────────────────────────────┘
```

The backend scripts load the persisted LightGBM model and SHAP values from `output/model/`, then serialize everything as static JSON. The React dashboard imports these JSON files at build time — no running Python server is needed to view the dashboard.

---

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Frontend dependencies
cd app/frontend
npm install
cd ../..
```

Requires [**uv**](https://docs.astral.sh/uv/) and **Node.js 18+**. Python 3.11+ is resolved automatically by `uv`.

## Run

### Using Make (recommended)

```bash
make all        # Run analysis + generate dashboard data
make serve      # Start dashboard dev server
make clean      # Remove all generated files
```

### Full pipeline (analysis + dashboard)

```bash
# 1. Run the analysis pipeline (generates output/)
uv run main.py              # modularized pipeline (preferred)
# uv run outlier_analysis.py  # original monolithic script (same output)

# 2. Generate dashboard data (reads output/, writes app/frontend/src/data/)
uv run app/backend/export_dashboard_data.py
uv run app/backend/whatif_simulator.py
uv run app/backend/reassignment_data.py

# 3. Start the interactive dashboard
cd app/frontend
npm run dev
# Opens at http://localhost:5173
```

### Dashboard only (if output/ already exists)

```bash
cd app/frontend
npm run dev
```

### Rebuild dashboard data (after re-running analysis)

```bash
uv run app/backend/export_dashboard_data.py
uv run app/backend/whatif_simulator.py
uv run app/backend/reassignment_data.py
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Key metrics (145 cases, 10.3% outlier rate, 98.6 min threshold), distribution histogram, top findings |
| **Physician Comparison** | Side-by-side stat cards, duration bar charts, radar chart of timing profiles, SHAP drivers per physician |
| **SHAP Explorer** | Interactive feature importance bars, toggle Global/Dr A/Dr B models, clinical tooltips, click-to-detail |
| **Outlier Deep Dive** | Sortable/filterable case table, expandable SHAP waterfall per case, physician and procedure type filters |
| **What-If Simulator** | Adjust procedure parameters with sliders, real-time outlier probability gauge, preset scenarios, SHAP contributions. *The What-If Simulator shows model behavior under hypothetical parameter changes. It is a demonstration tool for exploring feature sensitivities, not a clinical prediction system.* |
| **Reassignment** | Load any case, swap the physician, see outlier probability change. Batch reassignment scenarios and schedule optimizer to minimize total outliers. |
| **Trends** | Learning curves per physician, scheduling effects scatter plot, case complexity outlier rates |

---

## Output Directory Structure

```
output/
├── analysis_report.json                # Structured report (machine-readable)
├── analysis_summary.md                 # Full narrative summary (human-readable)
├── MSE433_M4_Data_with_outliers.csv    # Dataset with outlier label columns appended
├── eda/                                # Exploratory data analysis plots
│   ├── eda_distribution.png
│   ├── eda_outlier_classes.png
│   ├── eda_correlation.png
│   ├── eda_per_physician_distributions.png
│   ├── eda_per_physician_comparison.png
│   └── eda_per_physician_feature_comparison.png
├── global_model/                       # Global LightGBM + SHAP outputs
│   ├── lgbm_feature_importance.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_dependence.png
├── per_physician/                      # Per-physician models (only if >= 2 outliers)
│   ├── Dr_A/
│   └── Dr_B/
└── additional/                         # Supplementary analyses
    ├── learning_curve.png
    ├── case_complexity.png
    ├── case_order_scheduling.png
    └── physician_severity_profile.png
```

## Project Structure

```
MSE-433-Module-4/
├── app/
│   ├── frontend/                       # React 19 + TypeScript dashboard
│   │   ├── src/
│   │   │   ├── components/             # 8 page components + Layout
│   │   │   │   ├── Layout.tsx
│   │   │   │   ├── Overview.tsx
│   │   │   │   ├── PhysicianComparison.tsx
│   │   │   │   ├── ShapExplorer.tsx
│   │   │   │   ├── OutlierDeepDive.tsx
│   │   │   │   ├── WhatIfSimulator.tsx
│   │   │   │   ├── PatientReassignment.tsx
│   │   │   │   └── Trends.tsx
│   │   │   ├── data/                   # Generated JSON (from backend scripts)
│   │   │   │   ├── dashboard_data.json
│   │   │   │   ├── whatif_data.json
│   │   │   │   └── reassignment_data.json
│   │   │   ├── types/index.ts          # TypeScript interfaces
│   │   │   ├── App.tsx                 # Tab-based page router
│   │   │   └── main.tsx                # Entry point
│   │   ├── package.json
│   │   └── vite.config.ts
│   └── backend/                        # Python data export scripts
│       ├── export_dashboard_data.py    # Generates dashboard_data.json
│       ├── whatif_simulator.py         # Generates whatif_data.json
│       └── reassignment_data.py       # Generates reassignment_data.json
├── Data/                               # Raw dataset + definitions
│   ├── MSE433_M4_Data.xlsx
│   └── MSE433_M4_Definitions.pdf
├── Background/                         # Course materials
│   └── MSE433_M4_MedicalProcedure.pdf
├── src/                                # Modularized analysis pipeline (14 features)
│   ├── config.py                       # Constants, paths, RANDOM_STATE=42
│   ├── data_loader.py                  # Excel loading, cleaning (Phase 1)
│   ├── eda.py                          # EDA, distribution plots (Phase 2)
│   ├── feature_eng.py                  # Feature engineering (Phase 3)
│   ├── model.py                        # LightGBM + SHAP (Phases 4-5)
│   ├── per_physician.py                # Per-physician analysis (Phase 6)
│   ├── additional.py                   # Learning curves, complexity (Phase 7)
│   ├── export.py                       # CSV, JSON, markdown export (Phase 8)
│   └── viz.py                          # Shared viz constants
├── output/                             # Generated analysis outputs (gitignored)
│   └── model/                          # Persisted LightGBM model (global_model.pkl)
├── main.py                             # Entry point — runs src/ pipeline
├── outlier_analysis.py                 # Original monolithic script (reference)
├── Makefile                            # Build automation (make all/serve/clean)
├── requirements.txt                    # Pinned Python dependencies
├── .gitignore
└── README.md
```
