# Credit Risk Predictor (Streamlit)

A simple Streamlit app to predict the risk of a credit application using a trained XGBoost model (`xgboost_model.joblib`).

## Features
- Clean form with ideal low-risk defaults
- Probability, class prediction, and risk category
- Transparent view of model-ready feature vector
- Simple feature importance chart (if available)

## Files
- `app.py` — Streamlit UI
- `predict.py` — Model loader and feature vector builder
- `xgboost_model.joblib` — Trained XGBoost classifier (already in workspace)
- `requirements.txt` — Dependencies

## Run locally

```zsh
# (optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# start the app
streamlit run app.py
```

Then open the printed local URL (usually `http://localhost:8501`).

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repository (include `xgboost_model.joblib`).
2. Go to https://streamlit.io/cloud, sign in, and choose “New app”.
3. Select your repo and set the entry point to `app.py`.
4. Ensure `requirements.txt` is present.
5. Deploy and share the URL.

### Alternative deploy options
- Docker + any web host exposing port 8501
- Any platform that can run `streamlit run app.py` with Python 3.9+

## Notes
- The model expects one-hot encoded categorical features (drop-first applied during training). `predict.py` introspects the saved model to build the correct feature vector.
- If the model lacks stored feature names, `predict.py` falls back to a known schema for the public credit risk dataset; consider retraining/saving with `feature_names_in_` for maximum reliability.
