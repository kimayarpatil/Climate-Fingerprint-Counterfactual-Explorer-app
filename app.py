import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Climate AI", layout="wide")

# ---------------- CUSTOM CSS (🔥 UI MAGIC) ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #00d4ff;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_pickle("climate_final(1).pkl")

df = load_data()

with open("climate_model(1).pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🌍 Climate AI")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "📈 Trends",
    "🌡 Seasonal",
    "🌍 Global",
    "⚠️ Anomalies",
    "🤖 Prediction"
])

# ---------------- FILTER ----------------
st.sidebar.subheader("Filters")

start = st.sidebar.date_input("Start Date", df['Date'].min())
end = st.sidebar.date_input("End Date", df['Date'].max())

df = df[(df['Date'] >= str(start)) & (df['Date'] <= str(end))]

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.title("📊 Climate Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("🌡 Avg Temp", round(df['TAVG'].mean(),2))
    col2.metric("🔥 Max Temp", round(df['TAVG'].max(),2))
    col3.metric("❄️ Min Temp", round(df['TAVG'].min(),2))

    st.markdown("### 📈 Temperature Trend")
    fig = px.line(df, x='Date', y='TAVG')
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TRENDS ----------------
elif page == "📈 Trends":
    st.title("📈 Trend Analysis")

    fig = px.line(df, x='Date', y=['TAVG','Rolling_7','Rolling_30'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Distribution")
    fig2 = px.histogram(df, x='TAVG', nbins=50)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- SEASONAL ----------------
elif page == "🌡 Seasonal":
    st.title("🌡 Seasonal Analysis")

    month_avg = df.groupby('Month')['TAVG'].mean().reset_index()
    fig = px.bar(month_avg, x='Month', y='TAVG')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌈 Heatmap")
    heatmap = df.pivot_table(values='TAVG', index='Month', columns='Year')
    fig2 = px.imshow(heatmap, aspect='auto')
    st.plotly_chart(fig2)

# ---------------- GLOBAL ----------------
elif page == "🌍 Global":
    st.title("🌍 Global Comparison")

    fig = px.line(
        df,
        x='Date',
        y=['TAVG','TAVG_raw','land_and_ocean','global_tavg_monthly']
    )
    st.plotly_chart(fig)

# ---------------- ANOMALIES ----------------
elif page == "⚠️ Anomalies":
    st.title("⚠️ Extreme Events")

    df['Z'] = (df['TAVG'] - df['TAVG'].mean()) / df['TAVG'].std()
    anomalies = df[abs(df['Z']) > 2]

    st.write("Extreme Events:")
    st.dataframe(anomalies[['Date','TAVG']].head(20))

    fig = px.scatter(df, x='Date', y='TAVG', color=abs(df['Z']) > 2)
    st.plotly_chart(fig)

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":
    st.title("🤖 AI Temperature Predictor")

    col1, col2, col3 = st.columns(3)

    year = col1.number_input("Year", 2000, 2100, 2025)
    month = col2.slider("Month", 1, 12, 6)
    day = col3.slider("Day", 1, 31, 15)

    tmin = st.number_input("Min Temp", value=0.0)
    temp_range = st.number_input("Temp Range", value=1.0)

    # Auto features
    date = pd.Timestamp(year, month, day)
    day_of_year = date.dayofyear
    week = date.week
    quarter = (month-1)//3 + 1

    input_data = np.array([[
        year, month, day,
        day_of_year, week, quarter,
        tmin, temp_range, 5,
        0,0,0,
        0,0,
        0,0,
        0,0,0
    ]])

    if st.button("🚀 Predict"):
        pred = model.predict(input_data)
        st.success(f"🌡 Predicted Temperature: {round(pred[0],2)} °C")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✨ Built with Streamlit | Climate AI Project")
