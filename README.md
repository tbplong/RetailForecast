# RetailForecast — Demand Forecasting Engine

A hybrid demand forecasting pipeline that combines **XGBoost**, **Chronos-2** (Amazon's foundation time-series model), and a **meta-model ensemble** to produce accurate 168-hour (1-week) demand forecasts for retail products. The system includes a full training notebook, a Ray Serve API backend, and a Streamlit dashboard UI.

---

## Project Structure

| File | Description |
|---|---|
| `pipeline_final.ipynb` | End-to-end training pipeline: data loading, feature engineering, SAITS imputation, XGBoost/LightGBM/CatBoost training, deep learning models, Chronos-2 forecasting, and meta-model stacking |
| `serve_ensemble.py` | Ray Serve application that deploys XGBoost and Chronos-2 as microservices behind an ensemble router API |
| `app.py` | Streamlit dashboard for interactive forecast visualization with real-time evaluation metrics |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

- **Python** 3.10+
- **CUDA-capable GPU** (recommended for Chronos-2 and deep learning models)
- **Git** (to clone the repository)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/tbplong/RetailForecast.git
   cd RetailForecast
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Run the Training Pipeline

Open `pipeline_final.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially. The notebook walks through:

1. **Data Loading** — Downloads the [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) dataset from Hugging Face.
2. **Feature Engineering & Imputation** — Builds temporal features, applies SAITS neural imputation for missing demand values.
3. **Model Training** — Trains XGBoost (and optionally LightGBM, CatBoost, Ridge, KNN) regressors.
4. **Deep Learning Models** — Trains PyTorch-based time-series models.
5. **Chronos-2 Forecasting** — Runs Amazon's Chronos-2 foundation model for probabilistic forecasts.
6. **Meta-Model Stacking** — Combines XGBoost and Chronos-2 predictions via a learned ensemble.
7. **Evaluation** — Computes MAE, RMSE, WAPE, and Bias metrics across all models.

The notebook produces the following artifacts needed by the serving layer:

- `trained_model_xgboost.pkl` — Trained XGBoost model
- `meta_model_stacker.pkl` — Ensemble weights for the stacking meta-model
- `sample_xgb.pkl`, `sample_chronos_hist.pkl`, `sample_chronos_future.pkl` — Sample data for the Streamlit UI

### 2. Start the Ray Serve API

After training completes and the model artifacts are saved, launch the serving backend:

```bash
serve run serve_ensemble:app
```

This starts a local Ray Serve deployment at `http://127.0.0.1:8000` with:

- **XGBoostPredictor** — Serves XGBoost predictions
- **ChronosPredictor** — Serves Chronos-2 predictions (GPU-accelerated)
- **EnsembleRouter** — Orchestrates both models and returns blended forecasts via a `/forecast` POST endpoint

### 3. Launch the Streamlit Dashboard

In a separate terminal, run:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). From the dashboard you can:

- Select a store/product ID
- Generate a 168-hour forecast
- View XGBoost, Chronos, and Hybrid Ensemble predictions on a line chart
- See real-time evaluation metrics (MAE, RMSE, WAPE, Bias)
- Browse the hourly forecast breakdown table

> **Note:** The Ray Serve API (`serve run serve_ensemble:app`) must be running before using the dashboard.

---

## API Reference

### `POST /forecast`

**Request body (JSON):**

```json
{
  "xgb_features": { "feature1": [...], "feature2": [...] },
  "chronos_data": {
    "historical_data": { "id": [...], "timestamp": [...], "target": [...] },
    "future_data": { "id": [...], "timestamp": [...] }
  }
}
```

**Response:**

```json
{
  "status": "success",
  "ensemble_predictions": [0.12, 0.34, ...],
  "raw_xgboost": [0.10, 0.30, ...],
  "raw_chronos": [0.14, 0.38, ...]
}
```

---

## License

This project is provided as-is for educational and research purposes.
