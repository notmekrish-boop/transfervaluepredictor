# Premier League Transfer Value Predictor ⚽💰

An end-to-end Machine Learning project to predict English Premier League player transfer market values based on individual performance statistics and profile metrics.

Built with **Python**, **requests**, **pandas**, **scikit-learn**, and **matplotlib**.

---

## 📌 Project Overview

In the modern transfer market, player valuations can fluctuate wildly based on hype, club prestige, and media perception. This project builds an objective statistical benchmark using **Multiple Linear Regression** to determine what a player is fundamentally worth based purely on their season numbers:
- **Goals**
- **Assists**
- **Minutes Played**
- **Age**
- **Position** (Forward, Midfielder, Defender, Goalkeeper)

The model evaluates players against their actual Transfermarkt market value, flagging whether a star is **Underpriced (Bargain)**, **Overpriced (Carrying a Market/Hype Premium)**, or **Fairly Valued**.

---

## 🛠 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **`requests`** | REST API client for `football-data.org` (fetching scorers, squads, rate limit & auth handling) |
| **`pandas`** | Data manipulation, tabular multi-season cleaning, feature structuring |
| **`scikit-learn`** | Machine learning pipeline, `ColumnTransformer`, `OneHotEncoder`, `LinearRegression`, metrics |
| **`matplotlib`** | Visualizations (Actual vs Predicted, Feature Coefficients, Residuals, Metric distributions) |
| **`fastapi` & `uvicorn`** | High-performance backend REST API and live dashboard server |
| **Tailwind & Lucide** | Responsive Premier League dark-mode frontend dashboard with real-time sliders |
| **`joblib`** | Model persistence and quick inference reloading |

---

## 📂 Project Architecture

```
transfervaluepredictor/
│
├── config.py                   # Global constants, file paths, feature definitions, API configuration
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── premier_league_players.csv   # Curated multi-season dataset (2021-2024) with actual market values
│   └── raw/                         # Cached JSON API responses from football-data.org
│
├── src/
│   ├── __init__.py
│   ├── api_client.py           # requests-based FootballDataClient with caching & rate limit handling
│   ├── data_loader.py          # Data validation, filtering, and feature extraction
│   ├── model.py                # TransferValueModel wrapper (training, evaluation, explanations, persistence)
│   └── visualizer.py           # Matplotlib publication-grade plotting functions
│
├── models/
│   └── transfer_value_model.joblib  # Serialized trained scikit-learn pipeline
│
├── visualizations/
│   ├── actual_vs_predicted.png      # Scatter plot of Actual vs. Predicted with y=x diagonal
│   ├── feature_coefficients.png     # Learned weights bar chart (impact per goal, assist, year)
│   ├── residuals_distribution.png   # Error distribution with fitted normal curve
│   └── value_relationships.png      # Market Value vs Goals and Market Value vs Age plots
│
├── train.py                    # End-to-end training and evaluation pipeline
├── sync_live_data.py           # Live data synchronizer querying football-data.org API
├── predict.py                  # CLI inference tool (lookup existing players or custom stats)
├── demo.py                     # Non-interactive showcase across top Premier League stars
├── run_server.py               # Starts FastAPI server on http://localhost:8000
└── README.md                   # Complete documentation
```

---

## ⚡ Live Premier League Data Syncing

The application connects to **`football-data.org`** using your API token to pull current season live matchday scorers, matches played, and club crests.

To manually re-sync or pull fresh matchday data at any time:
```bash
python sync_live_data.py
```
This saves verified live players to `data/live_premier_league_players.json`, which the frontend and API serve automatically.

---

## 🧮 Mathematical Model: Multiple Linear Regression

The predictor estimates transfer market value $\hat{y}$ (in Millions EUR) as a linear combination of numerical statistics and one-hot encoded positions:

$$\hat{y} = \beta_0 + \beta_{\text{goals}} \cdot X_{\text{goals}} + \beta_{\text{assists}} \cdot X_{\text{assists}} + \beta_{\text{min}} \cdot X_{\text{minutes}} + \beta_{\text{age}} \cdot X_{\text{age}} + \sum_{p} \beta_{p} \cdot \mathbf{1}_{\{position = p\}}$$

### Learned Weights & Real-World Interpretation

From training on Premier League multi-season data:
- **Base Intercept ($\beta_0$)**: `+€119.92M`
- **Goals ($\beta_{\text{goals}}$)**: `+€3.27M` per goal scored in a season.
- **Assists ($\beta_{\text{assists}}$)**: `+€2.21M` per assist delivered.
- **Minutes Played ($\beta_{\text{min}}$)**: `+€4.37M` per 1,000 minutes on the pitch (rewarding durability & starting trust).
- **Age ($\beta_{\text{age}}$)**: `-€3.81M` per year of age (depreciation due to remaining contract/career length & resale value).

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have Python 3.10+ installed. Clone or navigate into the workspace and install requirements:

```bash
pip install -r requirements.txt
```

### 2. Optional: football-data.org API Token

The system operates **100% out-of-the-box offline** using the curated multi-season dataset. If you want to connect live to `football-data.org`, get a free API key at [football-data.org](https://www.football-data.org/client/register) and set the environment variable:

```powershell
$env:FOOTBALL_DATA_API_KEY="your_api_token_here"
```

### 3. Launch the Interactive Web Dashboard 🌐

Run the web server to start the frontend interface:

```bash
python run_server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

The dashboard includes:
1. **Scout Stars**: Live search autocomplete for 160+ Premier League players with side-by-side market value vs model value comparison and waterfall statistical decomposition.
2. **Custom Player Simulator**: Real-time interactive sliders for Age, Goals, Assists, Minutes, and Position with instant live valuation calculation.
3. **Players Database**: Filterable, sortable table across all Premier League clubs with one-click player inspection.
4. **Visualizations Hub**: High-resolution gallery of the 4 Matplotlib charts directly in your browser.

### 4. Train the Model & CLI Tools

```bash
python train.py
```

### 4. Run the Demo Showcase

Run a side-by-side comparison across top Premier League stars (Haaland, Palmer, Saka, Rice, Odegaard, Saliba, etc.):

```bash
python demo.py
```

---

## 🔍 Making Predictions (`predict.py`)

### A. Look up an existing Premier League star:
```bash
python predict.py --player "Erling Haaland"
python predict.py --player "Cole Palmer"
python predict.py --player "Bukayo Saka"
```

**Example Output:**
```
======================================================================
  TRANSFER VALUE ASSESSMENT: COLE PALMER (2023-24 SEASON)
======================================================================
 • Position:       Forward
 • Age:            22 years old
 • Goals Scored:   22
 • Assists:        11
 • Minutes Played: 2617 mins (29 matches equivalent)
----------------------------------------------------------------------

 MODEL PREDICTED VALUE:  €114.3M
 ACTUAL MARKET VALUE:     €80.0M
 VALUATION DIFFERENCE:   +34.3M (+42.9%)

 MARKET VERDICT:
  [BARGAIN / UNDERVALUED] The model estimates they are worth €34.3M MORE
  than their current price tag based on their output (high value potential!).

----------------------------------------------------------------------
 STATISTICAL DECOMPOSITION (HOW THE MODEL PRICED THIS PLAYER):
----------------------------------------------------------------------
   Base Intercept                                   : +119.92M
   Goals (22 × €3.27M)                              : +71.85M
   Assists (11 × €2.21M)                            : +24.36M
   Minutes Played (2617m × €0.0044M)                : +11.44M
   Age (22 yrs × €-3.81M)                           : -83.82M
   Position Premium (Forward)                       : -29.46M
   Raw Linear Model Output                          : €114.29M
   Final Estimated Value (min €0.5M)                : €114.3M
======================================================================
```

### B. Plug in custom stats for ANY player:
```bash
python predict.py --goals 15 --assists 8 --minutes 2700 --age 23 --position Forward
```

### C. Interactive Prompt Mode:
```bash
python predict.py --interactive
```

---

## 📊 Visualizations

All charts are saved in the `visualizations/` folder:

1. **`actual_vs_predicted.png`**: Scatter plot showing model predictions against actual market values with a $y=x$ ideal fit line.
2. **`feature_coefficients.png`**: Horizontal bar chart illustrating the direct impact of goals, assists, minutes, age, and position.
3. **`residuals_distribution.png`**: Residual error distribution with fitted normal curve.
4. **`value_relationships.png`**: Multi-panel relationships showing Market Value vs. Goals and Market Value vs. Age across positions.

---

## 🎓 Academic / Portfolio Highlights

- **Full Data-to-Inference Lifecycle**: API ingestion (`requests`) -> Preprocessing (`pandas`) -> Modeling & Pipelines (`scikit-learn`) -> Visualization (`matplotlib`).
- **Explainable AI (XAI)**: Every prediction is deconstructed into an exact algebraic decomposition showing what each goal, assist, and year of age contributed.
- **Robust Pipeline Design**: Uses `ColumnTransformer` and `Pipeline` to prevent data leakage during train/test splits.
