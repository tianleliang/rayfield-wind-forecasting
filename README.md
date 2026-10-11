# Wind Forecasting and Anomaly Engine

An open-source contribution, similar to what I built during my technical internship at Rayfield Systems (Harvard Undergraduate Ventures TECH), where I led an 8-intern team.

**What it does:** forecasts hourly wind power 336 hours (2 weeks) ahead from weather, then flags unusual hours with an Isolation Forest.

## Approach
1. **Future weather:** analogue bootstrap. Find the past week whose weather best matches the most recent week, and use what followed it as the forecast weather.
2. **Pass 1 (XGBoost):** weather → 6-hour rolling average of power.
3. **Pass 2 (XGBoost):** weather + pass 1's rolling average → hourly power.
4. **Anomaly detection:** Isolation Forest on physics-based features (power ÷ wind speed³, gustiness, sudden drops).

## Results

Each pass was first validated on its own. Pass 2 scored R² 0.94, but that test fed it the *actual* rolling average of recent power, so it measures pass 2 with ideal inputs, not the forecast. The honest number is end-to-end (weather → pass 1 → pass 2), scored against actual power on the last 20% of the data:

| Site | End-to-end R² | RMSE | Within ±0.10 |
|---|---|---|---|
| 1 | 0.72 | 0.159 | 50% |
| 2 | 0.12 | 0.168 | 43% |
| 3 | 0.12 | 0.278 | 21% |
| 4 | 0.74 | 0.115 | 70% |

**Why sites 2 and 3 fail:** The data-reporting and sites of loc 2 and 3 changed structurally. The wind-power correlation was stable from 2017 to 2019 (0.71–0.82), then fell at site 2 (0.53 by 2021) and collapsed at site 3 (0.35–0.41). Hours with near-zero power during high wind rose 3–6×, a sign of downtime or curtailment, and site 3 shifted to a different power curve. A weather-only model can't predict outages. Full diagnosis is in section 11 of the notebook.

**Next steps:** filter out non-operating hours before training, retrain site 3 on post-change data, and run the anomaly detector on observed data, where these outage hours are the anomalies worth catching.

## Data
The original Rayfield data is private. To run the notebook, use the similar public Kaggle dataset as toy data: [Wind Power Generation Data – Forecasting](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting) (4 locations, hourly, 2017–2021). Update the file paths in section 1.

## Run
Open `rayfield_wind_forecasting.ipynb` in Google Colab, point the paths at the 4 CSVs, and Run all (~5–10 min).

**Stack:** Python, XGBoost, scikit-learn, pandas, statsmodels, SHAP
