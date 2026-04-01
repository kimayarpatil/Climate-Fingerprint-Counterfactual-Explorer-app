# 🌍 Climate AI Dashboard

An interactive **AI-powered climate analytics dashboard** built using Streamlit.
This project analyzes historical climate data (1880–2025), detects anomalies, and provides predictions using machine learning.

---

## 🚀 Live Demo

👉 https://climate-fingerprint-counterfactual-explorer-app-z2ofxqwwbxoa.streamlit.app

---

## 📸 Dashboard Preview

* 🌡️ Temperature insights (Avg, Max, Min)
* 📊 Trend visualization
* 🔥 Climate risk indicator
* 📈 Interactive graphs
* ⚠️ Anomaly detection

---

## ✨ Features

### 📊 Data Visualization

* Temperature trend analysis (1880–2025)
* Interactive Plotly charts
* Animated climate change graphs
* Heatmaps with hover + zoom

### 🎛️ Filters (Sidebar)

* Date range selection
* Real-time filtering
* Dynamic updates

### 🤖 AI Prediction

* Machine learning model integration
* Climate trend forecasting
* User input-based predictions

### ⚠️ Anomaly Detection

* Detect unusual climate spikes
* Highlight extreme temperature events


## 🧠 Tech Stack

* Python
* Streamlit
* Pandas & NumPy
* Plotly
* Scikit-learn
* ReportLab (PDF generation)

---


## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/kimayarpatil/climate-figureprints-app.git
cd climate-figureprints-app
pip install -r requirements.txt
streamlit run app.py --server.port 8051
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Select repository
4. Set main file: `app.py`
5. Click Deploy

---

## ⚠️ Notes

* Ensure dataset has valid:

  * `Date`
  * `Year`
  * `Month`
  * `Temp`
* Avoid null values in temperature column
* Model files must exist or fallback logic should be used

---

## 🎯 Future Improvements

* 🌍 Map-based climate visualization
* 📡 Real-time weather API integration
* 🧠 Deep learning models (LSTM)
* 📱 Mobile UI optimization

---

## 👨‍💻 Author

GitHub: https://github.com/kimayarpatil

---

## ⭐ Support

If you like this project:

⭐ Star the repo
🍴 Fork it
📢 Share it

---

## 📌 License
This project is open-source and available under the MIT License.
