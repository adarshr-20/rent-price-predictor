import streamlit as st
import os
import requests
import compress_pickle
import joblib
import pandas as pd 
import numpy as np
from custom import Numerical_Feature_Adder, AvgPriceByLocalityAdder # Custom transformer

BEST_MODEL_URL = "https://github.com/adarshr-20/rent-price-predictor/releases/download/v1.0/best_model.gz"

def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url)
            if response.status_code != 200:
                st.error(f"Failed to download: {response.status_code}")
                st.stop()
            with open(filename, "wb") as f:
                f.write(response.content)
    return filename

@st.cache_resource
def load_model():
    model_path = download_file(BEST_MODEL_URL, "best_model.gz")
    try:
        model = compress_pickle.load(model_path, compression="gzip")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Page config
st.set_page_config(page_title="Rental Price Prediction", page_icon="🏙️", layout="centered")

# --- Title & Description ---
st.title("🏙️ Rental Price Prediction in Metropolitan Cities")
st.markdown("""
Welcome to the rental price predictor!  
Enter property details in the sidebar to estimate monthly rent.  
Press **Predict Rent** when ready.  
""")


model = load_model()

pipeline = joblib.load("full_pipeline.pkl")

# --- Sidebar Form ---
st.sidebar.header("🏠 Property Details")
with st.sidebar.form("prediction_form"):
    seller_type = st.selectbox("Seller Type", ["", "OWNER", "AGENT", "BUILDER"])
    bedroom = st.slider("Number of Bedrooms", 0, 10, 0)
    layout_type = st.selectbox("Layout Type", ["", "RK", "BHK"])
    property_type = st.selectbox("Property Type", ["", "Apartment", "Independent House", "Studio Apartment"])
    locality = st.text_input("Locality")
    area = st.number_input("Area (sq.ft.)", min_value=0.0, step=10.0)
    furnish_type = st.selectbox("Furnish Type", ["", "Furnished", "Semi-Furnished", "Unfurnished"])
    bathroom = st.slider("Number of Bathrooms", 0, 10, 0)
    city = st.selectbox("City", ["", "Mumbai", "Delhi", "Kolkata", "Bangalore", "Chennai", "Hyderabad", "Pune", "Ahmedabad"])
    
    submitted = st.form_submit_button("🔍 Predict Rent")

# --- On Submit ---
if submitted:
    if all(val in ["", 0, 0.0] for val in [seller_type, bedroom, layout_type, property_type, locality, area, furnish_type, bathroom, city]):
        st.error("🚨 Please fill at least one field before prediction.")
    else:
        data = pd.DataFrame([{
            "seller_type": seller_type if seller_type else None,
            "bedroom": bedroom if bedroom > 0 else None,
            "layout_type": layout_type if layout_type else None,
            "property_type": property_type if property_type else None,
            "locality": locality if locality else None,
            "area": area if area > 0 else None,
            "furnish_type": furnish_type if furnish_type else None,
            "bathroom": bathroom if bathroom > 0 else None,
            "city": city if city else None
        }])

        try:
            prepared_data = pipeline.transform(data)
            prediction = model.predict(prepared_data)
            predicted_price = np.expm1(prediction[0])

            # ✅ Output Prediction
            st.success(f"💰 Estimated Monthly Rent: ₹{predicted_price:,.0f}")
            st.balloons()

            # 🧾 Show Property Summary
            st.markdown("### 📋 Property Summary")
            st.dataframe(data.T.rename(columns={0: "Your Input"}))


        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


# ℹ️ Model Info
st.markdown("---")
with st.expander("📌 About this Model"):
    st.markdown("""  
    **Trained on**: [Kaggle Dataset](https://www.kaggle.com/datasets/saisaathvik/house-rent-prices-of-metropolitan-cities-in-india) [House Rent Prices of Metropolitan Cities in India]  
    🔔
                 **Note**: This model is built **for learning and educational purposes only**.  
                It is not meant for commercial or real-world deployment without further validation.
    """)


# 📌 Footer & Creator Info

with st.expander("👤 Creator Info"):
    st.markdown("""
    **Adarsh Rathore**  
    👨‍💻 Machine Learning Enthusiast  
    📧 [adarshrathore165@gmail.com](mailto:adarshrathore165@gmail.com) 
    🌐 [GitHub](https://github.com/adarshr-20) | [LinkedIn](https://www.linkedin.com/in/adarshr20/)

    *Thanks for using this app! If you found it useful, feel free to connect or drop feedback.* 😊
    """)

