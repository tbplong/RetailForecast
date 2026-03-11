import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="AI Forecast", page_icon="📈", layout="wide")
st.title("Demand Forecasting Engine")

@st.cache_data
def load_ui_data():
    try:
        xgb_df = pd.read_pickle("sample_xgb.pkl")
        chronos_hist = pd.read_pickle("sample_chronos_hist.pkl")
        chronos_future = pd.read_pickle("sample_chronos_future.pkl")
        return xgb_df, chronos_hist, chronos_future
    except FileNotFoundError:
        st.warning("Please save a small sample of your DataFrames to .pkl files so the UI can load them!")
        st.stop()

xgb_df, chronos_hist, chronos_future = load_ui_data()

st.sidebar.header("Forecast Settings")
# Get a list of unique products from the data
available_items = sorted(xgb_df['id'].unique())
selected_item = st.sidebar.selectbox("Select Store & Product ID:", available_items)

# ==========================================
# 4. THE API TRIGGER
# ==========================================
if st.button("Generate 168-Hour Forecast"):
    with st.spinner(f"Sending data to Ray Serve API for Item {selected_item}..."):
        item_xgb = xgb_df[xgb_df['id'] == selected_item].head(168)
        item_hist = chronos_hist[chronos_hist['id'] == selected_item]
        item_future = chronos_future[chronos_future['id'] == selected_item]
        
        def make_json_safe(df):
            temp_df = df.copy()
            datetime_cols = temp_df.select_dtypes(include=['datetime', 'datetimetz']).columns
            for col in datetime_cols:
                temp_df[col] = temp_df[col].astype(str)
            temp_df = temp_df.replace([np.inf, -np.inf, np.nan], None)
            return temp_df.to_dict(orient='list')

        api_xgb_features = item_xgb.drop(columns=['true_demand'], errors='ignore')

        payload = {
            "xgb_features": make_json_safe(api_xgb_features),
            "chronos_data": {
                "historical_data": make_json_safe(item_hist),
                "future_data": make_json_safe(item_future)
            }
        }

        try:
            response = requests.post("http://127.0.0.1:8000/forecast", json=payload)
            response.raise_for_status()
            result = response.json()
            
            st.success("Forecast Generated Successfully!")
            xgb_preds = np.clip(result['raw_xgboost'], a_min=0, a_max=None)
            chronos_preds = np.clip(result['raw_chronos'], a_min=0, a_max=None)
            ensemble_preds = np.clip(result['ensemble_predictions'], a_min=0, a_max=None)
            true_demand = item_xgb['true_demand'].values if 'true_demand' in item_xgb.columns else np.nan

            st.markdown("### 1-Week Inventory Ordering Requirement")
            
            total_xgb = np.sum(xgb_preds)
            total_chronos = np.sum(chronos_preds)
            total_ensemble = np.sum(ensemble_preds)
            total_demand = np.sum(true_demand) if not np.isnan(true_demand).all() else np.nan

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(label="XGBoost Estimate", value=f"{total_xgb:.4f} units")
            kpi2.metric(label="Chronos Estimate", value=f"{total_chronos:.4f} units")
            kpi3.metric(
                label="Recommended Order", 
                value=f"{total_ensemble:.4f} units",
            )
            kpi4.metric(
                label="True Demand (if available)", 
                value=f"{total_demand:.4f} units" if not np.isnan(total_demand) else "N/A",
            )
            
            st.divider()
            chart_data = pd.DataFrame({
                "0. True Demand (SAITS)": item_xgb['true_demand'].values if 'true_demand' in item_xgb.columns else np.nan,
                "1. Raw XGBoost": result['raw_xgboost'],
                "2. Raw Chronos": result['raw_chronos'],
                "3. Hybrid Ensemble": result['ensemble_predictions']
            })
            if 'date' in item_xgb.columns:
                chart_data.index = item_xgb['date'].values
            st.line_chart(chart_data, height=400, use_container_width=True)

            if 'true_demand' in item_xgb.columns and not item_xgb['true_demand'].isna().all():
                st.markdown("### Real-Time Evaluation Metrics")

                y_true = item_xgb['true_demand'].values
                valid_mask = ~np.isnan(y_true)
                y_true_clean = y_true[valid_mask]
                sum_true = np.sum(y_true_clean)
                
                metrics_list = []

                for model_name, col_name in [("XGBoost", "1. Raw XGBoost"), 
                                             ("Chronos", "2. Raw Chronos"), 
                                             ("Hybrid Ensemble", "3. Hybrid Ensemble")]:
                    
                    y_pred_clean = chart_data[col_name].values[valid_mask]
                    
                    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
                    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
                    
                    if sum_true > 0:
                        wape = np.sum(np.abs(y_true_clean - y_pred_clean)) / sum_true
                        bias = np.sum(y_pred_clean - y_true_clean) / sum_true
                    else:
                        wape, bias = np.nan, np.nan
                        
                    metrics_list.append({
                        "Model": model_name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "WAPE (%)": wape,
                        "Bias (%)": bias
                    })
                

                metrics_df = pd.DataFrame(metrics_list)
                
                styled_metrics = metrics_df.style.highlight_min(
                    subset=['MAE', 'RMSE', 'WAPE (%)'], color='#2e7d32', axis=0
                ).format({
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "WAPE (%)": "{:.2%}",
                    "Bias (%)": "{:+.2%}"
                })
                
                st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
                st.divider()
            st.markdown("### Hourly Forecast Breakdown")

            table_data = chart_data.reset_index().rename(columns={'index': 'Timestamp'})

            st.dataframe(
                table_data.style.format(precision=2, na_rep="N/A"), 
                use_container_width=True,
                hide_index=True
            )
            st.markdown("### Forecast Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### True Demand (SAITS Recovery)")

                st.line_chart(pd.DataFrame(item_xgb['true_demand'].values if 'true_demand' in item_xgb.columns else np.nan), height=250, use_container_width=True)
                
                st.markdown("##### Raw Chronos (Deep Learning)")
                st.line_chart(pd.DataFrame(result['raw_chronos']), height=250, use_container_width=True)
                
            with col2:
                st.markdown("##### Raw XGBoost (Machine Learning)")
                st.line_chart(pd.DataFrame(result['raw_xgboost']), height=250, use_container_width=True)
                
                st.markdown("##### Hybrid Ensemble (Final Output)")
                st.line_chart(pd.DataFrame(result['ensemble_predictions']), height=250, use_container_width=True)

            st.markdown("### Hourly Forecast Breakdown")
            
            table_data = chart_data.reset_index().rename(columns={'index': 'Timestamp'})
            
            st.dataframe(
                table_data.style.format(precision=2, na_rep="N/A"), 
                use_container_width=True,
                hide_index=True
            )
            with st.expander("View Raw API JSON Output"):
                st.json(result)
                
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Is your Ray Serve backend running on port 8000?")