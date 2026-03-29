import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Climate AI", layout="wide")

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Main */
.main {
    background: transparent;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
}

/* Text */
.title {
    font-size: 18px;
    color: #aaa;
}
.value {
    font-size: 36px;
    font-weight: bold;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00d4ff, #007cf0);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

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
st.sidebar.markdown("## 🌍 Climate AI Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "📈 Trends",
    "🌡 Seasonal",
    "🌍 Global",
    "⚠️ Anomalies",
    "🤖 Prediction"
])

# ---------------- FILTER ----------------
st.sidebar.subheader("📅 Filters")

start = st.sidebar.date_input("Start Date", df['Date'].min())
end = st.sidebar.date_input("End Date", df['Date'].max())

df = df[(df['Date'] >= str(start)) & (df['Date'] <= str(end))]

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.markdown("## 📊 Climate Dashboard")
    st.markdown("---")

    # Premium Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="title">🌡 Avg Temp</div>
            <div class="value">{round(df['TAVG'].mean(),2)}°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="title">🔥 Max Temp</div>
            <div class="value">{round(df['TAVG'].max(),2)}°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="title">❄️ Min Temp</div>
            <div class="value">{round(df['TAVG'].min(),2)}°C</div>
        </div>
        """, unsafe_allow_html=True)

    # AQI Style Card
    latest = df['TAVG'].iloc[-1]
    st.markdown(f"""
    <div class="card">
        <div class="title">🌍 Current Climate Status</div>
        <div class="value">{latest:.2f}°C</div>
        <div style="height:10px;border-radius:10px;
        background: linear-gradient(90deg, green, yellow, orange, red);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    st.markdown("### 📈 Temperature Trend")
    fig = px.line(df, x='Date', y='TAVG', template="plotly_dark", markers=True)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TRENDS ----------------
elif page == "📈 Trends":
    st.markdown("## 📈 Trend Analysis")
    st.markdown("---")

    fig = px.line(df, x='Date', y=['TAVG','Rolling_7','Rolling_30'],
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Distribution")
    fig2 = px.histogram(df, x='TAVG', nbins=50, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- SEASONAL ----------------
elif page == "🌡 Seasonal":
    st.markdown("## 🌡 Seasonal Analysis")
    st.markdown("---")

    month_avg = df.groupby('Month')['TAVG'].mean().reset_index()
    fig = px.bar(month_avg, x='Month', y='TAVG', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌈 Heatmap")
    heatmap = df.pivot_table(values='TAVG', index='Month', columns='Year')
    fig2 = px.imshow(heatmap, aspect='auto', template="plotly_dark")
    st.plotly_chart(fig2)

# ---------------- GLOBAL ----------------
elif page == "🌍 Global":
    st.markdown("## 🌍 Global Comparison")
    st.markdown("---")

    fig = px.line(
        df,
        x='Date',
        y=['TAVG','TAVG_raw','land_and_ocean','global_tavg_monthly'],
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- ANOMALIES ----------------
elif page == "⚠️ Anomalies":
    st.markdown("## ⚠️ Extreme Events")
    st.markdown("---")

    df['Z'] = (df['TAVG'] - df['TAVG'].mean()) / df['TAVG'].std()
    anomalies = df[abs(df['Z']) > 2]

    st.markdown("### 🚨 Extreme Events Table")
    st.dataframe(anomalies[['Date','TAVG']].head(20))

    fig = px.scatter(df, x='Date', y='TAVG',
                     color=abs(df['Z']) > 2,
                     template="plotly_dark")
    st.plotly_chart(fig)

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":
    st.markdown("## 🤖 AI Temperature Predictor")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    year = col1.number_input("Year", 2000, 2100, 2025)
    month = col2.slider("Month", 1, 12, 6)
    day = col3.slider("Day", 1, 31, 15)

    tmin = st.number_input("Min Temp", value=0.0)
    temp_range = st.number_input("Temp Range", value=1.0)

    date = pd.Timestamp(year, month, day)
    day_of_year = date.dayofyear
    week = date.isocalendar().week
    quarter = (month - 1)//3 + 1

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
