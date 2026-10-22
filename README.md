# EV Cars' Battery SoH Model

**Network Model URL:** http://10.101.58.94:8501/  
**Research Paper:** [Machine Learning-Based State-of-Health Estimation for Electric Car Batteries Using Real-World Driving Behaviour](https://ijsci.com/paper-details.php/79447773/machine-learning-based-state-of-health-estimation-for-electric-car-batteries-using-real-world-driving-behaviour)

An interactive Streamlit dashboard for EV battery State of Health (SoH) estimation using Python, machine learning, and two battery datasets.

## Features

- Explore both datasets and inspect missing values.
- Visualize SoH distributions, feature relationships, categorical comparisons, and correlations.
- Compare Extra Trees and Random Forest regression models.
- Evaluate MAE, RMSE, and R² with actual-vs-predicted and residual plots.
- Analyse permutation feature importance.
- Estimate SoH for custom vehicle records.
- Review illustrative battery-condition bands and data-quality signals.

## Project Structure

```text
EV_Battery_SoH_Working_Model/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── run_windows.bat
└── data/
    ├── ev_battery_degradation.csv
    └── ev_battery_failure_prediction.csv
```

## Run Locally

1. Install Python 3.10 or newer.
2. Extract the project ZIP.
3. Open a terminal in the folder containing `app.py` and `requirements.txt`.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

6. Open the URL displayed in the terminal, usually `http://localhost:8501`.

The CSV files are stored in the `data` folder, so no Google Drive mount is required.

## GitHub Upload

1. Create a repository on [GitHub](https://github.com/new).
2. Open the project folder in VS Code and start a terminal.
3. Run:

   ```powershell
   git init
   git add .
   git commit -m "Add EV Battery SoH Working Model"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ev-battery-soh-dashboard.git
   git push -u origin main
   ```

Replace `YOUR-USERNAME` with your GitHub username. Create the repository before pushing, and confirm that the datasets are permitted to be published.

## Deploy as a Website

GitHub stores the source code but does not run the Streamlit dashboard by itself. To publish the interactive application, connect the repository to [Streamlit Community Cloud](https://share.streamlit.io/), select the `main` branch and `app.py`, then deploy.

**Network note:** The `10.101.58.94:8501` address is a private-network address and may only work on the relevant network. Use your hosting provider's public URL when sharing the deployed dashboard externally.

## Research Notes

- Both datasets are processed separately because their schemas and target definitions differ.
- Records with missing target values are excluded from regression training.
- Identifiers, failure/status labels, and selected health-related fields are excluded from predictors where specified in the notebook.
- The dashboard provides research estimates, not certified battery diagnostics or automatic replacement decisions.
- Low test error does not establish real-world accuracy. Potential target leakage, dataset provenance, and validation on unseen vehicles should be investigated before making research claims.

## Troubleshooting

**`requirements.txt` not found:** Open the terminal in the folder that directly contains `requirements.txt` and `app.py`.

**Missing Python module:**

```powershell
python -m pip install -r requirements.txt
```

**Port already in use:**

```powershell
python -m streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

**Explore Changes error:** The scatter plot does not require the optional `statsmodels` package because its LOWESS trendline has been removed.

## Disclaimer

This project is intended for academic research and demonstration. Predictions depend on the supplied data and trained model and should not be used alone for battery safety, warranty, or replacement decisions.
