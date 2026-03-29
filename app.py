import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Climate AI", layout="wide")

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.title { color:#aaa; font-size:16px; }
.value { color:white; font-size:32px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_pickle("climate_final1.pkl")

df = load_data()

# ---------------- LOAD MODEL ----------------
with open("climate_model1.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Climate AI Dashboard")

page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "📈 Trends",
    "🌡 Seasonal",
    "🌍 Global",
    "⚠️ Anomalies",
    "🤖 Prediction"
])

# ---------------- FILTER ----------------
start = st.sidebar.date_input("Start Date", df['Date'].min())
end = st.sidebar.date_input("End Date", df['Date'].max())

df = df[(df['Date'] >= str(start)) & (df['Date'] <= str(end))]

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":

    st.title("📊 Climate Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Temp", round(df['TAVG'].mean(),2))
    col2.metric("Max Temp", round(df['TAVG'].max(),2))
    col3.metric("Min Temp", round(df['TAVG'].min(),2))

    # 🔥 Gauge Meter
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df['TAVG'].iloc[-1],
        title={'text': "Current Temperature"},
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 📈 Trend
    fig = px.line(df, x='Date', y='TAVG', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 🤖 AI Insight
    trend = df['TAVG'].iloc[-1] - df['TAVG'].iloc[0]
    if trend > 0:
        st.success("📈 AI Insight: Temperature is rising (Global Warming)")
    else:
        st.warning("📉 AI Insight: No strong warming trend")

# ---------------- TRENDS ----------------
elif page == "📈 Trends":

    st.title("📈 Trend Analysis")

    fig = px.line(df, x='Date', y=['TAVG','Rolling_7','Rolling_30'],
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 🔥 Correlation Heatmap
    st.subheader("🔥 Correlation Matrix")
    corr = df.corr(numeric_only=True)

    fig2 = px.imshow(corr, text_auto=True, template="plotly_dark")
    st.plotly_chart(fig2)

# ---------------- SEASONAL ----------------
elif page == "🌡 Seasonal":

    st.title("🌡 Seasonal Analysis")

    month_avg = df.groupby('Month')['TAVG'].mean().reset_index()

    fig = px.bar(month_avg, x='Month', y='TAVG', template="plotly_dark")
    st.plotly_chart(fig)

# ---------------- GLOBAL ----------------
elif page == "🌍 Global":

    st.title("🌍 Global Comparison")

    fig = px.line(df, x='Date',
                  y=['TAVG','land_and_ocean','global_tavg_monthly'],
                  template="plotly_dark")
    st.plotly_chart(fig)

# ---------------- ANOMALIES ----------------
elif page == "⚠️ Anomalies":

    st.title("⚠️ AI Anomaly Detection")

    model_if = IsolationForest(contamination=0.02)
    df['anomaly'] = model_if.fit_predict(df[['TAVG']])

    anomalies = df[df['anomaly'] == -1]

    st.write("🚨 Detected Anomalies")
    st.dataframe(anomalies[['Date','TAVG']].head(20))

    fig = px.scatter(df, x='Date', y='TAVG',
                     color=df['anomaly'],
                     template="plotly_dark")
    st.plotly_chart(fig)

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":

    st.title("🤖 AI Prediction & Forecast")

    col1, col2, col3 = st.columns(3)

    year = col1.number_input("Year", 2000, 2100, 2025)
    month = col2.slider("Month", 1, 12, 6)
    day = col3.slider("Day", 1, 31, 15)

    tmin = st.number_input("Min Temp", value=0.0)
    temp_range = st.number_input("Temp Range", value=1.0)

    date = pd.Timestamp(year, month, day)

    input_data = np.array([[year, month, day,
                            date.dayofyear, date.isocalendar().week,
                            (month-1)//3+1,
                            tmin, temp_range, 5,
                            0,0,0,0,0,0,0,0,0]])

    if st.button("🚀 Predict"):
        pred = model.predict(input_data)
        st.success(f"🌡 Predicted Temp: {round(pred[0],2)} °C")

    # 🔥 ARIMA Forecast
    st.subheader("📈 Future Forecast")

    model_arima = ARIMA(df['TAVG'], order=(5,1,0))
    model_fit = model_arima.fit()

    forecast = model_fit.forecast(steps=12)

    st.line_chart(forecast)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✨ Premium Climate AI Dashboard | Final Year Project 🚀")
