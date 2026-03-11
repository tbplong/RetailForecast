import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import pandas as pd
import numpy as np
import pickle
import torch
from chronos import Chronos2Pipeline
from fastapi import FastAPI, Request

# Create a FastAPI app instance right below your imports
api_app = FastAPI()
def calculate_dynamic_ensemble(xgb_preds, chronos_preds, xgb_features_dict):
    """
    Dynamically shifts ensemble weights based on hourly covariates.
    Assumes xgb_features_dict contains lists for 'is_promotion', 'holiday', or 'discount'.
    """
    xgb_arr = np.array(xgb_preds, dtype=float)
    chronos_arr = np.array(chronos_preds, dtype=float)

    is_promo = np.array(xgb_features_dict.get('is_promotion', [0]*len(xgb_arr)), dtype=float)
    holiday = np.array(xgb_features_dict.get('holiday', [0]*len(xgb_arr)), dtype=float)
    
    event_active = (is_promo > 0) | (holiday > 0)

    dynamic_preds = np.where(
        event_active,
        (xgb_arr * 0.60) + (chronos_arr * 0.40),
        (xgb_arr * 0.40) + (chronos_arr * 0.60)
    )
    
    return dynamic_preds.tolist()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.4})
class XGBoostPredictor:
    def __init__(self, model_path: str):
        print("Loading XGBoost Model from PKL...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
    def predict(self, payload: dict) -> list:
        df = pd.DataFrame(payload)
        cols_to_drop = [col for col in ['id', 'timestamp', 'date'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        preds = self.model.predict(df)
        preds = np.clip(preds, a_min=0, a_max=None)
        return preds.tolist()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.4, "num_gpus": 1})
class ChronosPredictor:
    def __init__(self):
        print("Loading Chronos-2 Foundation Model onto GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=self.device,
            torch_dtype=torch.float32
        )
        
    def predict(self, payload: dict) -> list:
        hist_ids = payload.get('historical_data', {}).get('id', [])
        future_ids = payload.get('future_data', {}).get('id', [])
        item_id = hist_ids[0] if len(hist_ids) > 0 else (future_ids[0] if len(future_ids) > 0 else "Unknown")
        
        try:
            df = pd.DataFrame(payload.get('historical_data', {}))
            if df.empty or len(df) < 3:
                print(f"Item {item_id} has no history (Ghost Store!). Falling back to 0.")
                return [0.0] * 168
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            df = df.set_index('timestamp').resample('1H').asfreq().reset_index()
            df['id'] = item_id
            df['target'] = df['target'].fillna(0)
            df = df.tail(672)
            for col in ['discount', 'holiday', 'is_promotion']:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
            future_df = None
            if 'future_data' in payload and payload['future_data']:
                raw_future = pd.DataFrame(payload['future_data'])
                raw_future['timestamp'] = pd.to_datetime(raw_future['timestamp'])
                
                last_hist_time = df['timestamp'].max()
                perfect_dates = pd.date_range(start=last_hist_time + pd.Timedelta(hours=1), periods=168, freq='1H')
                future_df = pd.DataFrame({'timestamp': perfect_dates, 'id': item_id})
                
                future_df = pd.merge(future_df, raw_future, on=['id', 'timestamp'], how='left')
                
                for col in ['discount', 'holiday', 'is_promotion']:
                    if col in future_df.columns:
                        future_df[col] = future_df[col].fillna(0).astype(int)
            with torch.no_grad():
                forecast_df = self.pipeline.predict_df(
                    df=df,
                    future_df=future_df,
                    prediction_length=168,
                    quantile_levels=[0.75], 
                    id_column="id",
                    timestamp_column="timestamp",
                    target="target"
                )
                
            preds = forecast_df['0.75'].clip(lower=0).round(2).tolist()
            
            del df, future_df, forecast_df
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return preds
            
        except Exception as e:
            print(f"CHRONOS ERROR on Item {item_id}: {str(e)}")
            return [0.0] * 168
@serve.deployment
@serve.ingress(api_app)
class EnsembleRouter:
    def __init__(self, xgb_handle: DeploymentHandle, chronos_handle: DeploymentHandle):
        self.xgb_handle = xgb_handle
        self.chronos_handle = chronos_handle
        print("Loading Level 2 Meta-Model Stacker...")
        with open("meta_model_stacker.pkl", "rb") as f:
            self.meta_model = pickle.load(f)

    @api_app.post("/forecast")
    async def get_forecast(self, request: Request):
        json_data = await request.json()
        self.weights = pickle.load(open("meta_model_stacker.pkl", "rb"))
        xgb_future = self.xgb_handle.predict.remote(json_data['xgb_features'])
        chronos_future = self.chronos_handle.predict.remote(json_data['chronos_data'])
        
        xgb_preds = await xgb_future
        chronos_preds = await chronos_future
        
        features_dict = json_data.get("xgb_features", {})
        
        meta_df = pd.DataFrame({
            'xgb_pred': xgb_preds,
            'chronos_pred': chronos_preds
        })
        
        if hasattr(self.meta_model, 'feature_names_in_'):
            expected_features = self.meta_model.feature_names_in_
            for feat in expected_features:
                if feat not in meta_df.columns:
                    meta_df[feat] = features_dict.get(feat, [0] * len(xgb_preds))
            meta_df = meta_df[expected_features]
        xgb_w = self.weights['xgb_weight']
        chronos_w = self.weights['chronos_weight']
        
        smart_ensemble_preds = (np.array(xgb_preds) * xgb_w) + (np.array(chronos_preds) * chronos_w)
        smart_ensemble_preds = smart_ensemble_preds.tolist()
        
        return {
            "status": "success",
            "ensemble_predictions": smart_ensemble_preds,
            "raw_xgboost": xgb_preds,
            "raw_chronos": chronos_preds
        }

xgb_deployment = XGBoostPredictor.bind("trained_model_xgboost.pkl")
chronos_deployment = ChronosPredictor.bind()

app = EnsembleRouter.bind(xgb_deployment, chronos_deployment)