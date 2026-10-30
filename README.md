# 🔋 EV Battery Health Predictor

Predicts a lithium-ion battery's **capacity**, **State of Health (SOH)** and **Remaining Useful Life (RUL)** from the measurements of a single discharge cycle, and serves the predictions in a small Streamlit web app.

🔗 **Live app:** [BatteryHealth](https://ev-battery-health-rxtyr4kezt79ceafe8jdw6.streamlit.app/)

![App screenshot](images/singlecycle.png)
![App screenshot](images/mutliplecycles.png)

## Why this project

An EV battery slowly loses capacity as it ages, and the usual way to measure that is a full capacity test, which is slow. This project asks: **can the health of a battery be estimated from the voltage, current, temperature and discharge time of one cycle?**

It builds on my electrical engineering background (EV and IoT projects) and applies a full data science workflow to real battery aging data: cleaning, exploration, modelling, explainability, deployment.

## Dataset

[NASA Prognostics Center of Excellence, Li-ion Battery Aging Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). 18650 lithium-ion cells (rated about 2 Ah) were charged and discharged in a lab until they wore out. I used four batteries: **B0005, B0006, B0007 and B0018**, with 636 discharge cycles in total.

For every discharge cycle I summarised the raw voltage, current and temperature readings into six features:

| Feature | Meaning |
|---|---|
| `avg_voltage`, `min_voltage` | Average and lowest terminal voltage |
| `avg_current` | Average discharge current |
| `avg_temp`, `max_temp` | Average and peak temperature |
| `discharge_time_s` | Length of the discharge |

**Targets:** capacity (Ah), SOH (%) = capacity ÷ starting capacity × 100, and RUL = cycles left until SOH reaches 80% (the usual EV end-of-life line).

![Capacity fade of all four batteries](images/all_batteries_capacity.png)

## Approach

| Step | Notebook | What I did |
|---|---|---|
| 1 | `step1_load_data.ipynb` | Read the NASA `.mat` files, kept discharge cycles, calculated SOH, plotted capacity fade |
| 2 | `step2_eda.ipynb` | Combined four batteries, checked data quality, explored features and correlations |
| 3a | `step3a_train_model.ipynb` | Trained Linear Regression, Random Forest and XGBoost to predict SOH |
| 3b | `step3b_target_comparison.ipynb` | Tested whether predicting capacity first works better than predicting SOH directly |
| 4 | `step4_shap_explain.ipynb` | Explained the model with SHAP |
| 5 | `step5_rul_prediction.ipynb` | Predicted Remaining Useful Life |
| 6 | `step6_streamlit_app.ipynb` | Saved the final models and built the app |
| 7 | `step7_extended_analysis.ipynb` | Time series forecasting, healthy/degraded classification, clustering and a hypothesis test |

**Honest evaluation.** I split by **battery**, not by random rows. Consecutive cycles of one battery are almost identical, so a random split would leak the answers. Models train on B0005, B0006 and B0007 and are tested on **B0018, which they never saw**. I also repeated the SOH comparison with every battery held out in turn.

## Results

### 1. Predicting SOH directly (held-out battery B0018)

| Model | MAE (SOH points) | RMSE | R² |
|---|---|---|---|
| Linear Regression | 5.32 | 5.36 | 0.58 |
| Random Forest | 2.13 | 2.80 | 0.89 |
| **XGBoost** | **1.88** | **2.32** | **0.92** |

![Predicted vs actual SOH](images/predicted_vs_actual.png)

The tree models follow the real curve, including the small bumps, but they **under-predict early life** (by about 3.5 SOH points over the first 20 cycles).

### 2. A better way to define the target

SOH is measured against each battery's own first cycle, so every battery starts at exactly 100%, even though the batteries really start at different capacities (B0006 starts at 2.04 Ah, the others at about 1.86 to 1.89 Ah). That blurs the labels. Predicting **capacity in Ah** first, and converting to SOH afterwards, fixes most of it:

| Target | MAE on B0018 (SOH points) | RMSE | Early-life bias (first 20 cycles) |
|---|---|---|---|
| SOH directly | 1.88 | 2.32 | −3.5 |
| **Capacity, then SOH** | **0.87** | **1.12** | **+0.4** |

Repeating this with each battery held out in turn:

| Held-out battery | SOH directly (MAE) | Capacity first (MAE) |
|---|---|---|
| B0005 | 1.26 | 1.69 |
| B0006 | 7.58 | 2.20 |
| B0007 | 1.92 | 1.07 |
| B0018 | 1.88 | 0.87 |
| **Average** | **3.16** | **1.46** |

The capacity-first method is clearly better on average, and especially for B0006, but it is **not better for every battery** (B0005 got slightly worse). The app uses this method.

![SOH directly vs capacity first](images/step3b_comparison.png)

### 3. Remaining Useful Life

| Features | Model | MAE (cycles) | RMSE | R² |
|---|---|---|---|---|
| Raw | Linear Regression | 10.95 | 15.23 | 0.51 |
| Raw | Random Forest | 13.32 | 17.21 | 0.37 |
| Raw | XGBoost | 12.84 | 15.92 | 0.46 |
| Smoothed | Linear Regression | 15.79 | 19.83 | 0.16 |
| Smoothed | Random Forest | 16.10 | 20.58 | 0.10 |
| Smoothed | XGBoost | 16.39 | 18.84 | 0.24 |

A naive baseline that assumes every battery lasts the average number of cycles of the others (about 95) is off by **20.3 cycles** on B0018, so the models do beat it, but only modestly. RUL is much harder than SOH: the models are noisy early in life and improve near end of life. Smoothing the features over several cycles made things **worse**, not better.

![Predicted vs actual RUL](images/rul_predicted_vs_actual.png)

### 4. What does the model rely on? (SHAP)

SHAP was run on the XGBoost model that predicts SOH directly (Step 4).

| Feature | Mean \|SHAP\| (SOH points) |
|---|---|
| `avg_voltage` | 5.73 |
| `avg_current` | 1.40 |
| `discharge_time_s` | 1.34 |
| `avg_temp` | 0.36 |
| `max_temp` | 0.29 |
| `min_voltage` | 0.13 |

Average voltage dominates, followed by current and discharge time. Removing `min_voltage` barely changes the error (1.88 to 1.89), while removing `avg_current` makes it worse (1.88 to 2.17).

![SHAP feature importance](images/shap_bar.png)

![SHAP beeswarm](images/shap_beeswarm.png)

### 5. Beyond SOH: forecasting, classification, clustering and a hypothesis test (Step 7)

Four more questions on the same data, with the same honest rules: classifiers are tested on a battery they have not seen, and forecasts are only scored on cycles that come after the data used to fit them.

**Time series forecasting.** A rolling-origin backtest: 41 forecasts of the next 20 cycles across the four batteries, scored in SOH points.

| Model | MAE (SOH points) | Better than naive by |
|---|---|---|
| Naive ("nothing changes") | 2.72 | n/a |
| Linear trend (last 30 cycles) | 1.70 | 38% |
| Holt (damped) | 1.71 | 37% |
| ARIMA(1,1,0) with drift | 1.66 | 39% |

The three real models are too close to call. Forecasting the **end of life** from cycle 50 is much weaker: every forecast that reaches 80% is too optimistic (the linear trend is off by about 22 cycles on average, ARIMA by about 58) and damped Holt never reaches it, because capacity fade speeds up later in life.

![Forecasts from cycle 80](images/ts_forecast.png)

**Classification: healthy (SOH >= 80%) vs degraded.** Leave-one-battery-out, averaged over the four folds:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority baseline | 0.564 | 0.00 | 0.00 | 0.00 | 0.500 |
| Logistic Regression | 0.914 | 0.82 | 0.99 | 0.89 | 0.997 |
| Random Forest | 0.900 | 0.92 | 0.86 | 0.88 | 0.997 |
| XGBoost | 0.887 | 0.97 | 0.80 | 0.87 | 0.996 |
| Logistic Regression, no `discharge_time_s` | 0.957 | 0.98 | 0.90 | 0.93 | 0.998 |
| XGBoost, no `discharge_time_s` | 0.875 | 0.74 | 0.63 | 0.67 | 0.996 |

The models rank cycles almost perfectly (ROC-AUC about 0.997), but a fixed cut-off transfers unevenly to a new battery: XGBoost misses about a quarter of degraded cycles, and XGBoost without `discharge_time_s` fails completely on B0007 (F1 0.00). With only four folds, the model ranking is indicative, not final.

![Confusion matrices](images/classification_confusion.png)

**Clustering.** K-Means finds no clear natural number of clusters (silhouette only creeps from 0.37 to 0.42 as k grows), so I used k = 3 as a description. The raw clusters mix battery identity (agreement 0.36) with ageing (0.29 against healthy/degraded). Scaling the features **within each battery** removes the battery effect (agreement about 0) and recovers healthy/degraded better (agreement 0.52, silhouette 0.49). That suggests comparing a pack against its own early-life baseline.

![Clusters in PCA space](images/clustering_pca.png)

**Hypothesis test.** H0: peak temperature is the same in healthy and degraded cycles. Degraded cycles ran about **1.3 °C hotter** (39.0 vs 40.3 °C, Cohen's d about 1.0; Welch t-test and Mann-Whitney both p < 0.001, also after a Bonferroni correction for the six features tested). But cycles from one battery are not independent, so those p-values are too small. The stricter battery-level paired test (n = 4) gives p = 0.042 for peak temperature, which does not survive the correction, although degraded cycles were hotter in **all four** batteries. So I claim that degraded cycles *tend* to run hotter, not that it is proven. At battery level only discharge time stays significant.

![Peak temperature](images/hypothesis_max_temp.png)

## The app

Built with Streamlit. It has three tabs:

- **Single cycle:** enter the six measurements and the battery's starting capacity (or pick an early, mid or late life example) and get the predicted capacity, SOH, remaining cycles and a health status. It warns when an input is outside the range the model was trained on.
- **Upload many cycles:** upload a CSV (or use the built-in sample) to get SOH and RUL curves and a downloadable table.
- **About:** how it works, and its limits.

The app's final models are trained on **all four batteries**, so their own accuracy cannot be measured. All accuracy numbers above come from the held-out tests.

## Limitations

- **Very small dataset:** four batteries from one lab. Treat this as a proof of concept, not a tool for real vehicles.
- **One test battery for most results.** The leave-one-battery-out check helps, but four batteries is still very little.
- **SOH depends on the starting capacity.** If the rated capacity (2.0 Ah) is used instead of the battery's real first-cycle capacity, every SOH value shifts. For B0018 that is about 7 points.
- **Discharge time is nearly the same information as capacity.** Inside each battery their correlation is about 0.98 to 0.99. A real vehicle rarely completes a full discharge, so the model is less practical than the lab numbers suggest.
- **RUL is weak,** especially early in a battery's life. The Step 7 forecasts from early life are also too optimistic about end of life.
- **Step 7 conclusions rest on four batteries,** so the clustering, classification and hypothesis-test results are indicative only.
- **Lab conditions, not real driving.**
- The 80% end-of-life line is a convention.

## Run it yourself

```bash
git clone https://github.com/vrshy/EV-Battery-Health.git
cd EV-Battery-Health
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To re-run the notebooks from the start, download the NASA battery data ([direct download](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)), unzip it, extract the first inner zip (`BatteryAgingARC-FY08Q4`), and copy `B0005.mat`, `B0006.mat`, `B0007.mat` and `B0018.mat` into the `data/` folder. Then run the notebooks in order. The processed table `all_batteries.csv` is already included, so notebooks 3 onwards work without the raw files. Step 7 needs a few more packages: `python -m pip install -r requirements-notebooks.txt`.

## Project structure

```
├── app.py                  Streamlit app
├── requirements.txt
├── requirements-notebooks.txt  Extra packages for the notebooks (statsmodels, SciPy, ...)
├── all_batteries.csv       Per-cycle features for the four batteries
├── sample_input.csv        Example input for the upload tab
├── models/                 Saved XGBoost models and app metadata
├── data/                   Place the NASA .mat files here (not included)
├── images/                 Plots used in this README
├── results/                Result tables (CSV)
└── step1 ... step7         Notebooks, in order
```

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, statsmodels, SciPy, Matplotlib, Streamlit.

## Data citation

B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## Author

**VARSHINI GANESAN**, B.E. Electrical and Electronics Engineering. Moving into data science and analytics.
[GitHub](https://github.com/vrshy) · [LinkedIn](https://www.linkedin.com/in/varshiganesan)
