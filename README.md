# EV Battery SoH Working Model
#### Network-Model-URL: (http://10.101.58.94:8501/)

### Research-Paper-Published: (https://ijsci.com/paper-details.php/79447773/machine-learning-based-state-of-health-estimation-for-electric-car-batteries-using-real-world-driving-behaviour)

This project turns the supplied EV battery research notebook and two CSV datasets into an interactive Streamlit dashboard.

## Features
- Browse both CSV datasets and inspect missing values.
- View an SoH condition-band chart with review signals and caveats.
- Plot SoH distributions, numeric-variable relationships, category comparisons, and correlations.
- Train and compare Extra Trees and Random Forest regression models.
- View actual-vs-predicted and residual plots.
- Calculate permutation feature importance.
- Enter a custom vehicle record and estimate SoH.

## Run locally
1. Install Python 3.10 or newer.
2. Extract this ZIP file.
3. Open a terminal in the extracted `EV_Battery_SoH_Working_Model` folder.
4. Install dependencies:

   `python -m pip install -r requirements.txt`

5. Start the dashboard:

   `streamlit run app.py`

6. Open the local URL printed in the terminal (usually http://localhost:8501) or (http://10.101.58.94:8501/).

The two CSV files are bundled in the `data` folder, so no Google Drive mount is needed.

## Important research notes
- The two datasets are kept separate because their schemas and target definitions differ.
- Target-missing records are excluded from regression training.
- IDs, failure/status labels, and several possible direct/derived health fields are excluded from predictors, following the supplied notebook's approach.
- Predictions are estimates for research demonstration, not certified diagnostics or automatic replacement decisions.
- The degradation dataset's very low test error should be interpreted cautiously. Validate data-generation assumptions and leakage risks before using the metric as a publication claim.


## Windows troubleshooting
- Run the commands from this extracted project folder, where `app.py` and `requirements.txt` are located.
- Install packages with the same Python interpreter used to run Streamlit: `python -m pip install -r requirements.txt`.
- Start the app with: `python -m streamlit run app.py`.
- The Explore Changes scatter chart intentionally does not use a LOWESS trendline, so `statsmodels` is not required.
- If a package is still missing, run `python -m pip install <package-name>` in the same terminal and restart Streamlit.
