import os
import joblib
import streamlit as st
import google.generativeai as genai
import pandas as pd


# Load trained model + metadata

@st.cache_resource
def load_artifacts():
    try:
        pipe = joblib.load("artifacts/best_model.joblib")
    except FileNotFoundError:
        st.error("❌ best_model.joblib not found in artifacts/. Run train.py first.")
        pipe = None

    try:
        meta = joblib.load("artifacts/inference_schema.joblib")
    except FileNotFoundError:
        st.error("❌ inference_schema.joblib not found in artifacts/. Run train.py first.")
        meta = None

    return pipe, meta


pipe, meta = load_artifacts()


# Configure Gemini

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
try:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    gemini_model = None
    st.error(f"⚠️ Gemini model not initialized: {e}")


# Streamlit UI

st.set_page_config(page_title="🚗 Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")
st.write("Enter car details below to estimate selling price...")

# Sidebar: Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("📜 Prediction History")
if st.session_state.history:
    for i, h in enumerate(st.session_state.history[::-1], 1):
        st.sidebar.markdown(f"**{i}. {h['brand']} ({h['year']})**")
        st.sidebar.write(f"💰 ₹ {h['price_inr']:,.0f} | 💵 ${h['price_usd']:,.0f}")
        st.sidebar.caption(f"ℹ️ {h['explanation'][:80]}...")

# Input form
with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year", min_value=1985, max_value=2025, value=2017)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=2000000, value=50000)
        mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=19.0)
        engine = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200)
        max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=500, value=82)

    with col2:
        seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8], index=2)
        fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
        brand = st.text_input("Brand (e.g., Maruti, Hyundai, Honda)", "Maruti")
        torque_nm = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=150)  # ✅ NEW

    submitted = st.form_submit_button("🔮 Predict Price")


# Prediction & Explanation

if submitted and pipe:
    # Prepare input row
    row = {
        "year": year,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel": fuel,
        "transmission": transmission,
        "seller_type": seller_type,
        "brand": brand,
        "car_age": 2025 - year,
        "torque_nm": torque_nm  
    }
    X_input = pd.DataFrame([row])

    # Prediction
    try:
        pred_inr = pipe.predict(X_input)[0]
        usd_rate = 0.012  # fixed INR→USD conversion (1 INR ≈ 0.012 USD)
        pred_usd = pred_inr * usd_rate

        st.success(f"💰 Estimated Price: ₹ {pred_inr:,.0f}  |  💵 ${pred_usd:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        pred_inr, pred_usd = None, None

    # Gemini explanation
    explanation = ""
    if pred_inr and gemini_model:
        prompt = f"""
        The predicted selling price for this car is ₹ {pred_inr:,.0f} (≈ ${pred_usd:,.0f}).
        Car details: {row}.
        Please explain in 3-4 sentences why the model might have predicted this price,
        considering age, mileage, brand reputation, and engine performance.
        """

        try:
            response = gemini_model.generate_content(prompt)
            explanation = response.text if response and response.text else "No explanation generated."
            st.info(f"🤖 Gemini Insight:\n\n{explanation}")
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            explanation = "Gemini explanation unavailable."

    # Save to history
    if pred_inr:
        st.session_state.history.append({
            "brand": brand,
            "year": year,
            "price_inr": pred_inr,
            "price_usd": pred_usd,
            "explanation": explanation
        })


# Optional: Chat with Gemini

st.markdown("---")
st.subheader("💬 Ask Gemini about Car Price Trends")

user_q = st.text_input("Your Question:")
if st.button("Ask Gemini"):
    if user_q.strip() and gemini_model:
        try:
            response = gemini_model.generate_content(user_q)
            st.write(response.text if response and response.text else "No answer generated.")
        except Exception as e:
            st.error(f"Gemini API error: {e}")
