# EV Battery SoH Working Model

**Research Paper:** [Machine Learning-Based State-of-Health Estimation for Electric Car Batteries Using Real-World Driving Behaviour](https://ijsci.com/paper-details.php/79447773/machine-learning-based-state-of-health-estimation-for-electric-car-batteries-using-real-world-driving-behaviour)

An interactive Streamlit dashboard for electric vehicle battery State of Health (SoH) estimation using Python, machine learning, and two CSV datasets.

## Features
- Explore datasets and missing values.
- Visualize SoH distributions, correlations, and feature relationships.
- Compare Extra Trees and Random Forest regression models.
- Evaluate model performance using MAE, RMSE, and R².
- View actual-versus-predicted and residual plots.
- Analyse permutation feature importance.
- Predict SoH for custom vehicle records.

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Project Structure

```text
EV_Battery_SoH_Working_Model/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── ev_battery_degradation.csv
    └── ev_battery_failure_prediction.csv
```

## Research Note

The datasets are analysed separately because their schemas and target definitions differ. Predictions are research estimates, not certified battery diagnostics. Model performance should be validated against independent data before making real-world claims.
