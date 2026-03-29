import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🌍 Climate AI Pro", layout="wide")

# ---------------- TITLE ----------------
st.title("🌍 Climate Fingerprint AI Explorer")
st.markdown("### 📊 Advanced Climate Analysis + Prediction Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Land_and_Ocean_complete.txt", delim_whitespace=True)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

view = st.sidebar.radio("Select Section", [
    "📊 EDA Dashboard",
    "📈 Trend Analysis",
    "🤖 Prediction",
    "🔍 Data Explorer"
])

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df.Year.min()),
    int(df.Year.max()),
    (1950, 2020)
)

filtered_df = df[(df.Year >= year_range[0]) & (df.Year <= year_range[1])]

# ---------------- EDA DASHBOARD ----------------
if view == "📊 EDA Dashboard":
    st.header("📊 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("📅 Total Records", len(filtered_df))
    col2.metric("🌡 Avg Temp Anomaly", round(filtered_df['LandOceanTemperatureIndex'].mean(), 3))
    col3.metric("📈 Max Temp", round(filtered_df['LandOceanTemperatureIndex'].max(), 3))

    st.subheader("📌 Temperature Distribution")
    fig = px.histogram(filtered_df, x="LandOceanTemperatureIndex", nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Yearly Average Trend")
    yearly = filtered_df.groupby("Year")["LandOceanTemperatureIndex"].mean().reset_index()
    fig2 = px.line(yearly, x="Year", y="LandOceanTemperatureIndex")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TREND ANALYSIS ----------------
elif view == "📈 Trend Analysis":
    st.header("📈 Climate Trend Analysis")

    rolling = st.slider("Select Rolling Window", 1, 24, 12)

    filtered_df['RollingAvg'] = filtered_df['LandOceanTemperatureIndex'].rolling(rolling).mean()

    fig = px.line(filtered_df, x="Date", y=["LandOceanTemperatureIndex", "RollingAvg"])
    st.plotly_chart(fig, use_container_width=True)

    st.info("Rolling average helps smooth fluctuations and show long-term trends.")

# ---------------- PREDICTION ----------------
elif view == "🤖 Prediction":
    st.header("🤖 Future Climate Prediction")

    degree = st.slider("Polynomial Degree", 1, 5, 2)

    X = df[['Year']]
    y = df['LandOceanTemperatureIndex']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    future_year = st.slider("Select Future Year", 2021, 2100, 2030)

    pred = model.predict(poly.transform([[future_year]]))[0]

    st.success(f"🌡 Predicted Temperature Anomaly for {future_year}: **{round(pred, 3)} °C**")

    # Plot prediction curve
    future_years = np.arange(1880, 2100).reshape(-1, 1)
    future_preds = model.predict(poly.transform(future_years))

    fig = px.line(x=future_years.flatten(), y=future_preds, labels={'x': 'Year', 'y': 'Temperature'})
    st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA EXPLORER ----------------
elif view == "🔍 Data Explorer":
    st.header("🔍 Raw Data Exploration")

    st.subheader("📄 Dataset Preview")
    st.dataframe(filtered_df.head(50))

    st.subheader("📊 Dataset Info")

    st.write("Shape:", filtered_df.shape)
    st.write("Columns:", list(filtered_df.columns))
    st.write("Data Types:")
    st.write(filtered_df.dtypes)

    st.subheader("❗ Missing Values")
    st.write(filtered_df.isnull().sum())

    st.subheader("🔁 Duplicate Rows")
    st.write(filtered_df.duplicated().sum())

    st.subheader("📅 Date Range")
    st.write("Min Date:", filtered_df['Date'].min())
    st.write("Max Date:", filtered_df['Date'].max())

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Climate AI Project")
