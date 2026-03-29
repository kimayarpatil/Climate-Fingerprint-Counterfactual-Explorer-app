import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Climate AI Explorer", layout="wide")

st.title("🌍 Climate Fingerprint AI Explorer")
st.markdown("### 📊 Advanced Climate Analysis + Prediction Dashboard")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, delim_whitespace=True)
    return df

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset (.txt)", type=["txt"]
)

use_sample = st.sidebar.checkbox("Use Default Dataset")

# ------------------ DATA HANDLING ------------------
if uploaded_file:
    df = load_data(uploaded_file)

elif use_sample:
    file_path = os.path.join("data", "Land_and_Ocean_complete.txt")
    
    if os.path.exists(file_path):
        df = load_data(file_path)
    else:
        st.error("❌ Default dataset not found. Please upload file.")
        st.stop()
else:
    st.warning("👈 Upload dataset or select default dataset")
    st.stop()

# ------------------ DATA PREPROCESS ------------------
df.columns = [
    "Year", "Month", "Decimal_Date",
    "Land_Temp", "Ocean_Temp",
    "Land_Ocean_Temp"
]

df["Time"] = df["Year"] + (df["Month"] / 12)

# ------------------ SIDEBAR FILTER ------------------
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (1900, 2020)
)

filtered_df = df[
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "📈 Trends", "🤖 Prediction", "📂 Data"
])

# ------------------ TAB 1: DASHBOARD ------------------
with tab1:
    st.subheader("🌡 Climate Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Land Temp", round(filtered_df["Land_Temp"].mean(), 2))
    col2.metric("Avg Ocean Temp", round(filtered_df["Ocean_Temp"].mean(), 2))
    col3.metric("Avg Combined", round(filtered_df["Land_Ocean_Temp"].mean(), 2))

    fig = px.line(
        filtered_df,
        x="Time",
        y="Land_Ocean_Temp",
        title="🌍 Global Temperature Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TAB 2: TRENDS ------------------
with tab2:
    st.subheader("📈 Temperature Trends")

    option = st.selectbox(
        "Select Temperature Type",
        ["Land_Temp", "Ocean_Temp", "Land_Ocean_Temp"]
    )

    fig = px.line(filtered_df, x="Time", y=option, title=f"{option} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(filtered_df, x=option, title="Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ TAB 3: PREDICTION ------------------
with tab3:
    st.subheader("🤖 Temperature Prediction")

    model = LinearRegression()

    X = filtered_df[["Time"]]
    y = filtered_df["Land_Ocean_Temp"]

    model.fit(X, y)

    future_year = st.slider("Select Future Year", 2021, 2100, 2030)

    future_time = future_year + (6 / 12)
    prediction = model.predict([[future_time]])

    st.success(f"🌡 Predicted Temperature in {future_year}: {prediction[0]:.2f} °C")

    # Plot prediction
    future_df = pd.DataFrame({
        "Time": [future_time],
        "Predicted": prediction
    })

    fig = px.scatter(future_df, x="Time", y="Predicted", title="Future Prediction")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TAB 4: DATA ------------------
with tab4:
    st.subheader("📂 Dataset Preview")
    st.dataframe(filtered_df)

    st.download_button(
        "⬇ Download Data",
        filtered_df.to_csv(index=False),
        file_name="filtered_data.csv"
    )
