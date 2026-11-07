
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load trained model
MODEL_PATH = "house_price_model.pkl"

st.title("🏠 House Price Prediction System")
st.markdown("### Machine Learning Based House Price Estimation")
st.write("Enter property details below to estimate its price.")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run python train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Sidebar
st.sidebar.title("🏡 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Predict House Price", "Model Performance"]
)

# Home page
if page == "Home":

    st.header("Welcome to House Price Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "545")
    col2.metric("Features", "12")
    col3.metric("Algorithms Trained", "3")

    st.subheader("About the Project")

    st.write("""
    This system uses Machine Learning regression algorithms
    to estimate house prices based on property characteristics.

    The trained models include:
    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
    """)

    st.info("Navigate to Predict House Price to try a prediction.")

# Prediction page
elif page == "Predict House Price":

    st.header("🏠 Enter House Details")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            area = st.number_input(
                "Area (square feet)",
                min_value=100,
                max_value=50000,
                value=3000
            )

            bedrooms = st.number_input(
                "Number of Bedrooms",
                min_value=1,
                max_value=10,
                value=3
            )

            bathrooms = st.number_input(
                "Number of Bathrooms",
                min_value=1,
                max_value=10,
                value=2
            )

            stories = st.number_input(
                "Number of Stories",
                min_value=1,
                max_value=10,
                value=2
            )

            parking = st.number_input(
                "Parking Spaces",
                min_value=0,
                max_value=10,
                value=1
            )

            mainroad = st.selectbox(
                "Main Road Access",
                ["yes", "no"]
            )

            guestroom = st.selectbox(
                "Guest Room",
                ["yes", "no"]
            )

        with col2:

            basement = st.selectbox(
                "Basement",
                ["yes", "no"]
            )

            hotwaterheating = st.selectbox(
                "Hot Water Heating",
                ["yes", "no"]
            )

            airconditioning = st.selectbox(
                "Air Conditioning",
                ["yes", "no"]
            )

            prefarea = st.selectbox(
                "Preferred Area",
                ["yes", "no"]
            )

            furnishingstatus = st.selectbox(
                "Furnishing Status",
                ["furnished", "semi-furnished", "unfurnished"]
            )

        submit = st.form_submit_button(
            "🔍 Predict House Price",
            type="primary",
            width="stretch"
        )

    if submit:

        input_data = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }])

        prediction = model.predict(input_data)[0]

        st.success("House price prediction completed!")

        st.subheader("Estimated House Price")

        st.metric(
            label="Predicted Price (Dataset Units)",
            value=f"{prediction:,.2f}"
        )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "market price. The dataset does not specify the currency "
            "or geographic location."
        )

# Model performance page
elif page == "Model Performance":

    st.header("📊 Model Performance Comparison")

    if os.path.exists("model_comparison.csv"):

        results = pd.read_csv("model_comparison.csv")

        st.dataframe(results, width="stretch")

        st.subheader("R2 Score Comparison")

        chart_data = results.set_index("Algorithm")["R2 Score"]

        st.bar_chart(chart_data)

    else:
        st.warning("Please run train.py to generate model results.")
