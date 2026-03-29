import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Climate AI Explorer", layout="wide")

# ------------------ CUSTOM CSS (🔥 PREMIUM UI) ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    padding: 10px;
    border-radius: 10px;
}
h1, h2, h3 {
    color: white;
}
.css-1d391kg {
    background-color: #111 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🌍 Climate Fingerprint AI Explorer")
st.markdown("### 📊 Advanced Climate Analysis + Prediction Dashboard")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=r"\s+", skiprows=4, engine="python")
    return df

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset (.txt)", type=["txt"])

# ------------------ DATA LOAD ------------------
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("👈 Please upload dataset")
    st.stop()

# ------------------ CLEAN DATA ------------------
df.columns = [
    "Year", "Month", "Decimal_Date",
    "Land_Temp", "Ocean_Temp",
    "Land_Ocean_Temp"
]

df["Time"] = df["Year"] + df["Month"]/12

# ------------------ FILTER ------------------
year_range = st.sidebar.slider(
    "📅 Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (1900, 2020)
)

filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "📈 Trends", "🔥 Insights", "🤖 Prediction", "📂 Data"]
)

# ------------------ DASHBOARD ------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("🌡 Avg Land Temp", round(filtered_df["Land_Temp"].mean(), 2))
    col2.metric("🌊 Avg Ocean Temp", round(filtered_df["Ocean_Temp"].mean(), 2))
    col3.metric("🌍 Combined Temp", round(filtered_df["Land_Ocean_Temp"].mean(), 2))

    fig = px.line(filtered_df, x="Time", y="Land_Ocean_Temp",
                  title="🌍 Global Temperature Trend",
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TRENDS ------------------
with tab2:
    option = st.selectbox("Select Feature", [
        "Land_Temp", "Ocean_Temp", "Land_Ocean_Temp"
    ])

    fig = px.line(filtered_df, x="Time", y=option,
                  title=f"{option} Over Time",
                  template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(filtered_df, x=option,
                        title="Distribution",
                        template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ INSIGHTS ------------------
with tab3:
    st.subheader("🔥 AI Insights")

    corr = filtered_df[[
        "Land_Temp", "Ocean_Temp", "Land_Ocean_Temp"
    ]].corr()

    fig = px.imshow(corr,
                    text_auto=True,
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)

    # Auto insight
    trend = filtered_df["Land_Ocean_Temp"].iloc[-1] - filtered_df["Land_Ocean_Temp"].iloc[0]

    if trend > 0:
        st.success("📈 Temperature is increasing over time (Global Warming)")
    else:
        st.warning("📉 No significant warming trend detected")

# ------------------ PREDICTION ------------------
with tab4:
    st.subheader("🤖 Future Prediction")

    model = LinearRegression()
    X = filtered_df[["Time"]]
    y = filtered_df["Land_Ocean_Temp"]
    model.fit(X, y)

    future_year = st.slider("Select Year", 2021, 2100, 2030)

    future_time = future_year + 0.5
    prediction = model.predict([[future_time]])

    st.success(f"🌡 Predicted Temp in {future_year}: {prediction[0]:.2f} °C")

    fig = px.scatter(x=[future_time], y=prediction,
                     title="Prediction Point",
                     template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ DATA ------------------
with tab5:
    st.dataframe(filtered_df)

    st.download_button(
        "⬇ Download CSV",
        filtered_df.to_csv(index=False),
        "data.csv"
    )
