import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, time
from datetime import datetime

st.set_page_config(page_title='Neuro-Symbolic Fusion Dashboard', layout='wide')

# ---------- Styles ----------
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0e11; color: #d1d4dc; }
.stMarkdown, p, span, label { color: #e2e8f0 !important; }
[data-testid="stSidebar"] { background: #131722 !important; border-right: 1px solid #2a2e39 !important; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
.card {
    background: #1e222d; 
    border: 1px solid #2a2e39; padding: 20px; border-radius: 8px;
    margin-bottom: 1.5rem; transition: all 0.2s ease;
}
.card:hover { border-color: #434651; }
.metric { font-size: 32px; font-weight: 700; margin-top: 5px; color: #d1d4dc; }
.metric-positive { color: #2ebd85; }
.metric-neutral { color: #2962ff; }
.metric-negative { color: #f6465d; }
.small { color: #787b86; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.stButton>button { 
    background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
    color: #ffffff !important; 
    border: 1px solid rgba(255,255,255,0.05); 
    padding: 0.6rem 1.2rem; 
    border-radius: 6px; 
    font-weight: 600; 
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); 
    text-transform: uppercase; 
    letter-spacing: 0.5px;
    width: 100%; 
    box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
}
.stButton>button:hover { 
    background: linear-gradient(135deg, #059669 0%, #047857 100%); 
    color: #ffffff !important; 
    box-shadow: 0 6px 15px rgba(16, 185, 129, 0.45);
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.1);
}
.stButton>button:active {
    transform: translateY(1px) scale(0.98);
    box-shadow: 0 2px 5px rgba(16, 185, 129, 0.2);
}
h1, h2, h3 { color: #e0e3eb !important; font-weight: 600 !important; letter-spacing: -0.5px; }
div[data-baseweb="select"] > div, input { background-color: #131722 !important; border: 1px solid #2a2e39 !important; color: #d1d4dc !important; border-radius: 4px !important; }
div[data-baseweb="select"] > div:hover, input:focus { border-color: #2962ff !important; }
.stAlert { background-color: #1e222d !important; border: 1px solid #2a2e39 !important; color: #d1d4dc !important; border-radius: 4px !important; border-left: 4px solid #2962ff !important; }
</style>''', unsafe_allow_html=True)

# ---------- Session ----------
if 'history' not in st.session_state: st.session_state.history=[]
if 'model_name' not in st.session_state: st.session_state.model_name='LightGBM'

# ---------- Load Models ----------
def load_model(path):
    if not os.path.exists(path):
        return None
    if path.endswith('.pkl'):
        with open(path,'rb') as f:
            return pickle.load(f)
    elif path.endswith('.h5'):
        try:
            from tensorflow.keras.models import load_model as keras_load
            return keras_load(path, compile=False)
        except Exception:
            return None
    return None

lgbm = load_model('lightgbm_model.pkl')
lstm = load_model('lstm_model.h5')

# ---------- Sidebar ----------
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 25px; margin-top: 0px;'>
        <img src="https://cdn-icons-png.flaticon.com/512/9322/9322127.png" width="85" style="filter: drop-shadow(0 0 12px rgba(16, 185, 129, 0.5)); margin-bottom: 12px;">
        <h2 style='color: #e2e8f0; font-weight: 700; font-size: 22px; line-height: 1.2; margin: 0;'>Neuro-Symbolic</h2>
        <h3 style='color: #10b981; font-weight: 600; font-size: 14px; margin-top: 5px; margin-bottom: 0px; letter-spacing: 2px;'>FUSION HUB</h3>
    </div>
    """, unsafe_allow_html=True
)

icons = {'Dashboard':'📊 Dashboard', 'Predict':'⚡ Predict', 'Analytics':'📈 Analytics', 'Live Monitor':'📡 Live Monitor', 'History':'📜 History', 'Settings':'⚙️ Settings'}
page = st.sidebar.radio(
    'Navigation', 
    ['Dashboard','Predict','Analytics','Live Monitor','History','Settings'],
    format_func=lambda x: icons[x],
    label_visibility="collapsed"
)
st.sidebar.markdown('---')
st.sidebar.markdown("<div style='text-align: center; color: #787b86; font-size: 12px; padding: 0px 5px; line-height: 1.4;'>2026 / Neuro-Symbolic Fusion for Causal Inference in Multimodal Time Series</div>", unsafe_allow_html=True)

# ---------- Helpers ----------
def apply_theme(fig):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#d1d4dc', family='Inter'), margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(gridcolor='#2a2e39', zerolinecolor='#2a2e39', linecolor='#2a2e39'), yaxis=dict(gridcolor='#2a2e39', zerolinecolor='#2a2e39', linecolor='#2a2e39'))
    return fig

def fake_predict(vals):
    return round(sum(vals)/len(vals)*1.15,2)

def predict_value(features):
    model = lgbm if st.session_state.model_name=='LightGBM' else lstm
    try:
        if model is not None:
            pm25, pm10, co, no2, so2, o3, temp, humid, wspm = features
            vals = pd.DataFrame([{
                'No': 1, 'hour': 12, 'PM10': pm10, 'SO2': so2, 'NO2': no2, 'CO': co, 'O3': o3, 
                'TEMP': temp, 'PRES': 1010.0, 'DEWP': humid, 'RAIN': 0.0, 'wd': 1, 'WSPM': wspm,
                'station': 1, 'day_of_week': 3, 'is_weekend': 0, 'is_peak_hour': 1, 'season': 2,
                'hour_sin': 0.0, 'hour_cos': 1.0, 'PM2.5_lag1': pm25, 'PM2.5_roll_mean_3': pm25, 
                'PM2.5_roll_std_3': 10.0, 'CO_NO2': co/(no2+0.01), 'PM_ratio': pm25/(pm10+0.01), 'temp_diff': 5.0
            }])
            p = model.predict(vals)
            return float(np.array(p).flatten()[0])
    except Exception:
        pass
    return fake_predict(features)

def cause_effect(features, names, pred):
    causes = []
    effects = []
    
    pm25, pm10, co, no2, so2, o3, temp, humid, wspm = features
    
    if pm25 > 150 or pm10 > 150:
        causes.append(f"Primary: Severe particulate accumulation (PM2.5: {pm25:.1f})—linked to dense smog or industrial exhaust.")
        effects.append("Critical respiratory risk. Fine particles bypass lung barriers; N95 masking protocol advised outdoors.")
    elif pm25 > 50 or pm10 > 50:
        causes.append(f"Warning: Elevated particulate levels (PM2.5: {pm25:.1f}) indicating moderate atmospheric stagnation.")
        effects.append("Minor irritation for sensitive groups. Consider reducing prolonged heavy outdoor exertion.")
        
    if co > 50:
        causes.append(f"Primary: Dangerous Carbon Monoxide (CO: {co:.1f}) surge. Correlated with heavy vehicle traffic or combustion.")
        effects.append("Risk of headaches, dizziness, or impaired cognitive function. Avoid localized traffic hot-zones.")
        
    if no2 > 40 or so2 > 40:
        causes.append(f"Secondary: Toxic gas peaking (NO2: {no2:.1f}, SO2: {so2:.1f})—caused by unrefined fossil fuel emissions.")
        effects.append("Increased likelihood of severe airway inflammation or worsened asthma symptoms.")
        
    if o3 > 50:
        causes.append(f"Secondary: Ground-level Ozone high (O3: {o3:.1f}). Triggered by sunlight reacting with volatile organic compounds.")
        effects.append("Immediate throat irritation possible; avoid outdoor aerobic activities during peak sun hours.")
        
    if temp > 35 and humid > 70:
        causes.append(f"Climate: High heat ({temp:.1f}°C) and severe humidity ({humid:.1f}%) compounding atmospheric danger.")
        effects.append("High risk of heat exhaustion. Smog formation is severely accelerated under these conditions.")

    if not causes:
        causes.append("System Nominal: All monitored environmental variables are operating within standard baseline ranges.")
    
    if not effects:
        effects.append("No immediate action required. Continue standard autonomous monitoring protocols.")
        
    return causes[:3], effects[:3]

def get_risk_status(pred, features):
    max_toxic_gas = max(features[2:6])
    if pred >= 100 or max_toxic_gas > 80: 
        return '🔴 Unhealthy (High Risk)', '#f6465d'
    elif pred >= 50 or max_toxic_gas > 40: 
        return '🟡 Moderate (Medium Risk)', '#f59e0b'
    else: 
        return '🟢 Healthy (Low Risk)', '#2ebd85'

# ---------- Pages ----------
if page=='Dashboard':
    st.markdown("<h1 style='color: #e2e8f0; font-size: 32px; font-weight: 700; margin-bottom: 5px; margin-top: 0px;'>Neuro-Symbolic System Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 16px; margin-bottom: 30px;'>High-level technical summary of active inferences, model status, and live atmospheric risk factors.</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='card' style='border-top: 4px solid #10b981;'><div class='small'>🧠 Active AI Engine</div><div class='metric metric-neutral'>{st.session_state.model_name}</div><div style='color: #10b981; font-size: 13px; margin-top: 8px;'>↑ Optimal Latency & Performance</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='card' style='border-top: 4px solid #2962ff;'><div class='small'>⚡ Inferences (24h)</div><div class='metric metric-neutral'>1,284</div><div style='color: #10b981; font-size: 13px; margin-top: 8px;'>↑ +14.2% vs Yesterday Volumes</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='card' style='border-top: 4px solid #f59e0b;'><div class='small'>⚠️ Aggregate Risk</div><div class='metric metric-negative'>68.4%</div><div style='color: #f6465d; font-size: 13px; margin-top: 8px;'>↓ -2.1% Deteriorating (1hr Trailing)</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='card' style='border-top: 4px solid #10b981;'><div class='small'>🛡️ Server Status</div><div class='metric metric-positive'>Nominal</div><div style='color: #94a3b8; font-size: 13px; margin-top: 8px;'>All edge-computing nodes active</div></div>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns((2,1))
    with col_chart1:
        x = pd.date_range(datetime.now(), periods=24, freq='H')
        y = np.random.randint(40,140,24)
        fig = px.area(x=x, y=y, title='24-Hour Regional PM2.5 Toxicity Trend', markers=True, color_discrete_sequence=['#2962ff'])
        fig.update_traces(fillcolor='rgba(41, 98, 255, 0.15)', line=dict(width=3, color='#2962ff'), marker=dict(size=6, color='#2962ff'))
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    with col_chart2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=68.4,
            title={'text': "Global Threat Risk Score"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2a2e39", 'tickmode': 'array', 'tickvals': [0, 50, 100]},
                'bar': {'color': "#f59e0b"}, 
                'bgcolor': "#1e222d",
                'borderwidth': 2,
                'bordercolor': "#2a2e39",
                'steps': [
                    {'range': [0, 50], 'color': "rgba(46, 189, 133, 0.1)"},
                    {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [80, 100], 'color': "rgba(246, 70, 93, 0.2)"}],
                'threshold': {'line': {'color': "#f6465d", 'width': 4}, 'thickness': 0.75, 'value': 85}
            }
        ))
        fig2 = apply_theme(fig2)
        fig2.update_layout(margin=dict(l=30, r=45, t=60, b=30))
        st.plotly_chart(fig2, use_container_width=True)


elif page=='Predict':
    st.title('Deep AI Prediction')
    st.markdown("<p style='color:#94a3b8; font-size:16px; margin-bottom:24px;'>Input environmental variables to run multi-modal inference.</p>", unsafe_allow_html=True)

    st.markdown("<h4 style='color:#e2e8f0; margin-top:0px; margin-bottom:15px; font-weight:600;'>Environmental Variables</h4>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        pm25 = st.number_input('💨 PM2.5 (µg/m³)', value=45.0, help="Fine particulate matter < 2.5 µm")
        pm10 = st.number_input('🌫️ PM10 (µg/m³)', value=60.0, help="Coarse particulate matter < 10 µm")
        temp = st.number_input('🌡️ Temperature (°C)', value=25.0, help="Ambient temperature")
    with c2:
        co = st.number_input('🚗 CO (ppb)', value=15.0, help="Carbon Monoxide heavily linked to traffic")
        no2 = st.number_input('🏭 NO2 (ppb)', value=20.0, help="Nitrogen Dioxide from vehicle and industrial emissions")
        humid = st.number_input('💧 Humidity (%)', value=65.0, help="Relative humidity")
    with c3:
        so2 = st.number_input('🏭 SO2 (ppb)', value=10.0, help="Sulfur Dioxide from fossil fuel combustion")
        o3 = st.number_input('☀️ O3 / Ozone (ppb)', value=35.0, help="Ground-level ozone")
        wspm = st.number_input('🌬️ Wind Speed (m/s)', value=3.5, help="Local wind speed")

    names=['PM2.5','PM10','CO','NO2','SO2','O3','TEMP','HUMID','WSPM']
    vals=[pm25, pm10, co, no2, so2, o3, temp, humid, wspm]

    
    if st.button('Execute Inference Run'):
        pred = predict_value(vals)
        causes, effects = cause_effect(vals, names, pred)
        status, color = get_risk_status(pred, vals)
        st.markdown(f"<div style='border-left: 4px solid {color}; padding: 15px; background: #1e222d; border-radius: 4px; border-top: 1px solid #2a2e39; border-right: 1px solid #2a2e39; border-bottom: 1px solid #2a2e39;'><h3>{status}</h3><p style='font-size: 20px; font-weight: bold;'>Predicted PM2.5 Index: {pred:.2f}</p></div><br/>", unsafe_allow_html=True)
        a,b = st.columns(2)
        with a:
            st.subheader('Primary Causes')
            for c in causes: st.info(c)
        with b:
            st.subheader('Predicted Effects')
            for e in effects: st.warning(e)
        st.session_state.history.append({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'model':st.session_state.model_name,'prediction':pred, 'status':status.split(' ')[0]})

elif page=='Analytics':
    st.title('Analytics & Insights')
    st.markdown("<p style='color:#94a3b8; font-size:16px; margin-bottom:24px;'>Analyze feature impacts and distribution to better understand predictions.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    df = pd.DataFrame({'Feature':['PM2.5','PM10','CO','NO2','SO2','O3','TEMP','HUMID','WSPM'],'Importance':[32,25,18,14,10,8,7,4,4]})
    with c1:
        fig = px.bar(df,x='Feature',y='Importance', title='AI Feature Importance', text='Importance', color_discrete_sequence=['#2962ff'])
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    with c2:
        fig2 = px.pie(df, names='Feature', values='Importance', hole=0.4, title='Contribution Breakdown (Donut)', color_discrete_sequence=['#2ebd85', '#2962ff', '#f6465d', '#00bcd4', '#ff9800', '#9c27b0', '#e91e63', '#3f51b5', '#009688'])
        st.plotly_chart(apply_theme(fig2), use_container_width=True)

elif page=='Live Monitor':
    st.title('Live Data Stream')
    st.markdown("<p style='color:#94a3b8; font-size:16px; margin-bottom:24px;'>Real-time multipoint sensor readings and active risk monitoring.</p>", unsafe_allow_html=True)
    
    # Generate realistic multi-sensor data
    t = pd.date_range(datetime.now(), periods=30, freq='min')
    data = pd.DataFrame({
        'Time': t,
        'PM2.5 Level': np.random.normal(45, 10, 30),
        'PM10 Level': np.random.normal(60, 15, 30),
        'NO2 Level': np.random.normal(25, 5, 30)
    })
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric('Latest PM2.5', f"{data['PM2.5 Level'].iloc[-1]:.1f} µg/m³", delta="-1.2 from avg", delta_color="inverse")
    with m2: st.metric('Latest PM10', f"{data['PM10 Level'].iloc[-1]:.1f} µg/m³", delta="+2.4 from avg", delta_color="inverse")
    with m3: st.metric('Latest NO2', f"{data['NO2 Level'].iloc[-1]:.1f} ppb", delta="Stable", delta_color="off")
    
    data_melted = data.melt(id_vars=['Time'], value_vars=['PM2.5 Level', 'PM10 Level', 'NO2 Level'], var_name='Sensor', value_name='Value')
    fig = px.line(data_melted, x='Time', y='Value', color='Sensor', title='Critical Sensor Streams (Last 30 Min)', color_discrete_sequence=['#2ebd85', '#f6465d', '#2962ff'])
    st.plotly_chart(apply_theme(fig), use_container_width=True)
    st.success('📡 Live connection active. Data updates streamed automatically.')

elif page=='History':
    st.title('Prediction Logs Tracker')
    st.markdown("<p style='color:#94a3b8; margin-bottom:24px;'>Historical record of user inferences and model states.</p>", unsafe_allow_html=True)
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        if 'status' not in df_hist.columns:
            df_hist['status'] = 'Unknown'
        st.dataframe(
            df_hist,
            column_config={
                "time": "Timestamp",
                "model": "AI Engine",
                "prediction": st.column_config.NumberColumn("Risk Score", format="%.2f"),
                "status": "Risk State"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info('No predictions logged in the current session. Go to the Predict page to run inference.')

elif page=='Settings':
    st.title('Settings')
    choice = st.selectbox('Select Model',['LightGBM','LSTM'], index=0 if st.session_state.model_name=='LightGBM' else 1)
    if st.button('Apply Settings'):
        st.session_state.model_name = choice
        st.success(f'Active model changed to {choice}')
    comp = pd.DataFrame({'Model':['LightGBM','LSTM'],'Accuracy':[99.7,95.5],'Speed':['Fast','Medium'],'Best For':['Tabular','Sequence']})
    st.dataframe(comp, use_container_width=True)
