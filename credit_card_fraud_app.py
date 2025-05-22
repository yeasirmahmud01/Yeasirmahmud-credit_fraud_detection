import streamlit as st
import joblib
import pandas as pd

# ─── Page Config & Style ─────────────────────────────────────────────
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.markdown("""
<style>
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
body {
    background-color: #f4f6f8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #333;
}
h1 {
    color: #007bff;
    margin-bottom: 0.2rem;
}
.instructions {
    max-width: 600px;
    margin: 0 auto 1.5rem auto;
    font-size: 16px;
    color: #555;
    line-height: 1.4;
    background: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
}
.card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-top: 1.5rem;
    text-align: center;
}
.stButton > button {
    background-color: #007bff;
    color: white;
    font-size: 16px;
    padding: 0.6rem 1.4rem;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s ease;
    margin-top: 1rem;
    width: 100%;
}
.stButton > button:hover {
    background-color: #0056b3;
    cursor: pointer;
}
.result-box {
    margin-top: 1.5rem;
    padding: 1.5rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 22px;
    user-select: none;
}
.fraud {
    background-color: #fddede;
    color: #b00020;
    border: 2px solid #b00020;
}
.normal {
    background-color: #dbf3e0;
    color: #1b5e20;
    border: 2px solid #1b5e20;
}
.summary {
    margin-top: 1rem;
    font-size: 18px;
    font-weight: 600;
    color: #444;
}
.footer {
    position: fixed;
    bottom: 0;
    width: 100vw;
    background: #232526;
    background: linear-gradient(90deg, #232526 0%, #414345 100%);
    color: white;
    text-align: center;
    padding: 0.6rem;
    font-size: 14px;
    box-shadow: 0 -2px 12px rgba(0,0,0,0.15);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.footer a {
    color: #ff9800;
    text-decoration: none;
    margin: 0 0.5rem;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ─── Header & Instructions ───────────────────────────────
st.markdown("<h1 style='text-align:center;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="instructions">
  <p><b>Upload a CSV file with these columns:</b></p>
  <ul>
    <li><b>Time</b>: Seconds since first transaction</li>
    <li><b>V1 to V28</b>: 28 anonymized features</li>
    <li><b>Amount</b>: Transaction amount</li>
  </ul>
  <p>This data will be used to detect possible fraudulent transactions.</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model and Scaler ───────────────────────────────
@st.cache_data
def load_model_and_scaler():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ─── File uploader ───────────────────────────────────────
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"CSV missing required columns: {missing_cols}")
    else:
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        if st.button("Predict"):
            with st.spinner("Detecting fraud..."):
                try:
                    X = df[required_cols].copy()
                    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
                    probs = model.predict_proba(X)[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    fraud_count = preds.sum()
                    total = len(preds)
                    avg_prob = probs.mean()

                    verdict = "Fraud Detected ⚠️" if fraud_count > 0 else "No Fraud Detected ✅"
                    verdict_class = "fraud" if fraud_count > 0 else "normal"

                    st.markdown(f"<div class='card'><h3>🔍 Prediction Results</h3></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='result-box {verdict_class}'>{verdict}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"""
                        <div class="summary">
                        Fraudulent Transactions: <b>{fraud_count}</b> / {total}<br>
                        Average Fraud Probability: <b>{avg_prob:.4f}</b>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ─── Footer ─────────────────────────────────────────────
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100vw;
    background: #1a1a1a;
    color: #bbb;
    text-align: center;
    padding: 0.4rem 1rem;
    font-size: 13px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    border-top: 1px solid #333;
    display: flex;
    justify-content: center;
    gap: 1.2rem;
}
.footer a {
    color: #58a6ff;
    text-decoration: none;
    transition: color 0.25s ease;
    font-weight: 500;
}
.footer a:hover {
    color: #1e90ff;
    text-decoration: underline;
}
</style>
<div class="footer">
    <div>Contact: <a href="mailto:yeasir.mahmud2503@gmail.com">yeasir.mahmud2503@gmail.com</a></div>
    <div><a href="https://github.com/yeasirmahmud01/Yeasirmahmud-credit_fraud_detection" target="_blank">GitHub</a></div>
    <div>Credits: <b>Yeasir Mahmud</b></div>
</div>
""", unsafe_allow_html=True)

