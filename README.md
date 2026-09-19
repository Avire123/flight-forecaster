# Flight Forecaster

A Streamlit-based flight fare trend forecaster for African airlines and international routes, with Kenyan shilling (KSh) pricing and model-based trend prediction.

## Features

- Synthetic airfare dataset modelled around African hub airports and international destinations
- Forecasting logic for buy-now vs wait recommendations
- Route trend visualizations with Plotly
- Airline comparison charts
- Optional Playwright live scraper for live flight listings
- Regression and classification modelling with a sklearn-compatible fallback

## Project structure

- `app.py` – Streamlit dashboard
- `model_pipeline.py` – data generation and forecasting pipeline
- `scraper.py` – Playwright live scrape utility
- `tests/test_flight_forecaster.py` – regression test
- `pytest.ini` – pytest configuration

## Running locally

1. Create and activate a virtual environment if needed.
2. Install dependencies:

   ```bash
   pip install streamlit pandas numpy plotly scikit-learn pytest
   ```

3. If you want to enable live scraping, also install Playwright and browser binaries:

   ```bash
   pip install playwright
   python -m playwright install chromium
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Usage

Select an African origin airport and an international destination, adjust the travel window, and review the model recommendation for whether to book now or wait.

## Notes

This project uses a synthetic fare dataset by default. The live scraper is optional and requires a compatible browser environment.
