# EV Battery SoH Working Model
**Published Research Paper:** [Machine Learning-Based State-of-Health Estimation for Electric Car Batteries Using Real-World Driving Behaviour](https://ijsci.com/paper-details.php/79447773/machine-learning-based-state-of-health-estimation-for-electric-car-batteries-using-real-world-driving-behaviour)

**Network URL:** (http://10.110.66.15:8501)

An interactive Streamlit dashboard for electric-vehicle battery State of Health (SoH) estimation using Python, machine learning, and two CSV datasets.

## Features

- Explore both datasets and inspect missing values.
- Visualize SoH distributions, feature relationships, category comparisons, and correlations.
- Compare Extra Trees and Random Forest regression models.
- Evaluate models using MAE, RMSE, and R².
- View actual-versus-predicted and residual plots.
- Analyse permutation feature importance.
- Estimate SoH for a custom vehicle record.

## Project Structure

```text
EV_Battery_SoH_Working_Model/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── ev_battery_degradation.csv
    └── ev_battery_failure_prediction.csv
```

## Run on Windows

### Step 1: Open the project folder

Extract the project ZIP. Open the project folder in VS Code, or open Command Prompt.

If using Command Prompt, navigate to the folder where you extracted the project. Replace the example path with your actual path:

```cmd
cd ".\EV_Battery_SoH_Model"
```

### Step 2: Check the current folder

Run:

```cmd
dir
```

Make sure the output includes `app.py`, `requirements.txt`, and the `data` folder.

**If `requirements.txt` is not found:** you are probably in the parent folder. Use `cd` to enter the inner project folder, then run `dir` again:

```cmd
cd EV_Battery_SoH_Working_Model
dir
```

Only run this extra `cd` command if the inner folder exists. Your terminal must be in the folder containing `app.py` and `requirements.txt`.

### Step 3: Install dependencies

Run this command from the correct project folder:

```cmd
python -m pip install -r requirements.txt
```

### Step 4: Start the dashboard

```cmd
python -m streamlit run app.py
```

### Step 5: Open the dashboard

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

To stop the dashboard, press `Ctrl+C` in the terminal.

## Troubleshooting

**`requirements.txt` not found**

Run `dir` to list the current folder contents. If `requirements.txt` and `app.py` are not listed, use `cd` to enter the correct project folder and try again.

**`No module named streamlit` or another missing package**

Run the dependency installation command again using the same Python interpreter that starts Streamlit:

```cmd
python -m pip install -r requirements.txt
```

**Port 8501 is already in use**

```cmd
python -m streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

**Explore Changes chart error**

The scatter chart does not use a LOWESS trendline, so the optional `statsmodels` package is not required.

## Research Notes

- The two datasets are processed separately because their schemas and target definitions differ.
- Records with missing target values are excluded from regression training.
- Identifiers, failure/status labels, and selected health-related fields are excluded from predictors where specified in the project notebook.
- Predictions are research estimates, not certified battery diagnostics or automatic replacement recommendations.
- Strong test metrics do not prove real-world accuracy. Dataset provenance, potential target leakage, and validation on unseen vehicles should be reviewed before making real-world claims.
