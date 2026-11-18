import streamlit as st
import joblib
import pandas as pd

# Load trained model
package = joblib.load("models/best_model.pkl")

model = package["model"]
features = package["features"]

# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Application heading
st.title("🚗 Used Car Price Prediction")

st.write(
    "Predict the estimated resale price of a "
    "used car using Machine Learning."
)

st.divider()

# Input fields
st.subheader("Enter Car Details")

car_name = st.text_input(
    "Car Name",
    "Hyundai i20"
)

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Manufacturing Year",
        min_value=1980,
        max_value=2026,
        value=2018,
        step=1
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=45000,
        step=1000
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    )

with col2:
    seller_type = st.selectbox(
        "Seller Type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Previous Ownership",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

st.divider()

# Prediction button
if st.button(
    "Predict Car Price",
    use_container_width=True
):

    if not car_name.strip():
        st.error("Please enter a car name.")

    else:

        input_data = {
            "name": car_name,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner
        }

        input_df = pd.DataFrame([
            {
                feature: input_data[feature]
                for feature in features
            }
        ])

        prediction = model.predict(input_df)[0]

        st.success("Prediction completed!")

        st.metric(
            "Estimated Selling Price",
            f"₹{prediction:,.2f}"
        )

        st.caption(
            "This is an estimated price based on historical "
            "training data, not a guaranteed market offer."
        )

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn "
    "and Random Forest Regression."
)
