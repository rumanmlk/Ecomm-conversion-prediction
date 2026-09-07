# E-commerce conversion prediction

Predict whether a session is likely to convert to a **purchase** from tabular event features. Two models, one Streamlit app.

Training used a **500k-row sample** of the [Kaggle multi-category store events](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) (October 2019). The demo compares a Random Forest pipeline with a small neural net.

## What it does

- Manual feature entry or CSV upload
- Side-by-side probabilities: **Random Forest** vs **neural net**
- Sample rows in the repo: `sampled_3_rows_with_purchase.csv`, `sampled_3_rows_no_purchase.csv`

The Streamlit app is the public demo. Training lives in the notebooks.

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit UI |
| `E_comm_final.ipynb` | Final EDA, features, training, evaluation |
| `E_comm.ipynb` | Earlier exploration |
| `models/full_pipeline.joblib` | Fitted RF pipeline |
| `models/final_nn_model.keras` | Fitted neural net |
| `models/feature_names.joblib` | Column order expected at inference |
| `requirements.txt` | Python deps |

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
