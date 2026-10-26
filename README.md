# 🔋 EV Battery Health Prediction

A machine-learning project that estimates **battery capacity, State of Health (SOH), and Remaining Useful Life (RUL)** from measurements collected during a lithium-ion battery discharge cycle. It includes exploratory data analysis, model comparisons, SHAP explainability, and an interactive **Streamlit** web app.

> **Project status:** Proof of concept using NASA laboratory battery-aging data. It is not validated for real-world vehicle or safety-critical use.

## ✨ Features

- Predicts battery capacity in ampere-hours (Ah).
- Converts predicted capacity into State of Health (SOH).
- Estimates Remaining Useful Life (RUL) in discharge cycles.
- Supports single-cycle predictions and batch CSV uploads.
- Displays battery-health status and warns when inputs are outside the training-data range.
- Explains model behaviour with SHAP feature-importance visualisations.
- Includes notebooks and result files for exploring the workflow.

## 🖥️ App preview

Add or keep screenshots in the `images/` folder and update these paths if you rename them.

![Streamlit app](images/app1.png)

![Streamlit app results](images/app2.png)

## 🧠 How it works

1. **Load data** — read NASA battery-aging `.mat` files.
2. **Prepare features** — summarise each discharge cycle using voltage, current, temperature, and discharge duration.
3. **Explore the data** — inspect capacity fade, distributions, and feature relationships.
4. **Train and compare models** — evaluate Linear Regression, Random Forest, and XGBoost.
5. **Explain predictions** — use SHAP to understand which features influence the model.
6. **Predict RUL** — compare models for estimating remaining discharge cycles.
7. **Run the app** — load saved models in Streamlit and make single-cycle or batch predictions.

### Input features

| Feature | Description | Unit |
|---|---|---|
| `avg_voltage` | Average terminal voltage during discharge | V |
| `min_voltage` | Minimum terminal voltage during discharge | V |
| `avg_current` | Average discharge current | A |
| `avg_temp` | Average battery temperature | °C |
| `max_temp` | Maximum battery temperature | °C |
| `discharge_time_s` | Duration of the discharge cycle | seconds |

The cycle number is not used as a model feature.

## 📊 Results

### State of Health (SOH)

The following comparison comes from the held-out-battery experiment for battery **B0018**. Metrics are in SOH percentage points.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Linear Regression | 5.32 | 5.36 | 0.584 |
| Random Forest | 2.13 | 2.80 | 0.886 |
| XGBoost | 1.88 | 2.32 | 0.922 |

A comparison of two prediction strategies found that predicting capacity first and then converting it to SOH performed better on average across the leave-one-battery-out experiments. The Streamlit app uses this capacity-first approach.

![SOH strategy comparison](images/step3b_comparison.png)

### Remaining Useful Life (RUL)

RUL is more difficult to estimate than SOH. On held-out battery B0018, the best result in the reported comparison was Linear Regression using raw features.

| Features | Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---:|---:|---:|
| Raw | Linear Regression | 10.95 cycles | 15.23 cycles | 0.505 |
| Raw | Random Forest | 13.32 cycles | 17.21 cycles | 0.368 |
| Raw | XGBoost | 12.84 cycles | 15.92 cycles | 0.459 |

The app uses a saved XGBoost model for RUL, so the RUL scores above should not be interpreted as the performance of that deployed model.

![RUL prediction comparison](images/rul_predicted_vs_actual.png)

### Model explainability

SHAP analysis in the experiments identified `avg_voltage` as the strongest contributor to SOH predictions.

![SHAP feature importance](images/shap_beeswarm.png)

## 🗂️ Project structure

```text
.
├── app.py
├── requirements.txt
├── all_batteries.csv
├── data/
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   ├── B0018.mat
│   └── README.txt
├── models/
│   ├── capacity_model.json
│   ├── rul_model.json
│   └── meta.json
├── results/
│   ├── model_results.csv
│   ├── rul_model_results.csv
│   └── ...
├── images/
│   └── plots and app screenshots
├── step1_load_data.ipynb
├── step2_eda.ipynb
├── step3a_train_model.ipynb
├── step3b_target_comparison.ipynb
├── step4_shap_explain.ipynb
├── step5_rul_prediction.ipynb
└── step6_streamlit_app.ipynb
```

## 🚀 Run the app locally

### 1. Clone the repository

Replace `YOUR-REPOSITORY-URL` with your GitHub repository URL.

```bash
git clone YOUR-REPOSITORY-URL
cd YOUR-REPOSITORY-FOLDER
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

### Reproduce the notebooks

Run the notebooks in order, from `step1_load_data.ipynb` through `step6_streamlit_app.ipynb`. The analysis notebooks may also require `scipy`, `matplotlib`, `seaborn`, and `shap`, which are not all included in the app's `requirements.txt`.

## 🧪 Using the app

### Single cycle
Enter the six discharge measurements and the battery's starting capacity. The app displays predicted capacity, SOH, RUL, and a health-status label. Example presets are available to try the interface.

### Multiple cycles
Upload a CSV containing one row per discharge cycle and the six required feature columns listed above. An optional `cycle` column can be used as the chart axis. The app returns predictions and supports downloading the results.

**Important:** SOH is calculated as:

`SOH (%) = predicted capacity (Ah) / starting capacity (Ah) × 100`

Use the battery's actual first-cycle capacity when available. The app's 80% SOH threshold is a conventional end-of-life indicator, not a universal rule for every battery chemistry or application.

## 🧰 Tech stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualisation:** Matplotlib, Seaborn
- **App:** Streamlit
- **Notebook environment:** Jupyter

## 📚 Dataset and acknowledgement

This project uses the NASA Ames Prognostics Center of Excellence lithium-ion battery-aging dataset, containing repeated charge, discharge, and impedance cycles for batteries **B0005, B0006, B0007, and B0018**.

**Reference:** B. Saha and K. Goebel (2007), *Battery Data Set*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, California.

Dataset information: [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## ⚠️ Limitations

- The dataset contains only four laboratory-tested batteries; results may not generalise to other cells or operating conditions.
- The saved app models are trained on all four batteries. The reported evaluation metrics come from separate held-out experiments and are not a direct held-out score for the saved app models.
- Accurate SOH depends on entering a suitable starting capacity.
- Discharge duration is strongly related to capacity in this dataset, while a full discharge may not be available during normal vehicle operation.
- RUL predictions are less reliable than SOH predictions and depend on the limited features and data used here.
- Laboratory cycling data does not represent every real-world driving, charging, temperature, or load condition.

**Do not use these predictions alone to make battery-safety, maintenance, or vehicle-operation decisions.**

## 🔮 Future improvements

- Test on more batteries and additional datasets.
- Estimate SOH from partial-discharge measurements.
- Use sequences of recent cycles to improve RUL estimation.
- Validate models under varied temperatures, charging patterns, and load profiles.
- Deploy a hosted demo with documented monitoring and validation.

## 👩‍💻 Author

**Varshini G**  
GitHub: [@vrshy](https://github.com/vrshy)  
LinkedIn: [Varshini Ganesan](https://linkedin.com/in/varshiganesan)
