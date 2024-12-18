import streamlit as st
import pandas as pd
import pickle
from preprocessing.cleaning_data import preprocess  # Import your preprocess function
from predict.prediction import predict


# Load the model
@st.cache_resource  
def load_model():
    model_path = "models/random_forest.pkl"  # Adjust path as needed
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model path is correct.")
        return None
    return model


# Load the income DataFrame
@st.cache_resource
def load_income_data():
    income_path = r"cleaned_income_data.csv"
    try:
        income_df = pd.read_csv(income_path)
        income_df = income_df[income_df["zipCode"] != 0]
        income_df = income_df.sort_values(by="zipCode", ascending=True)
        return income_df
    except FileNotFoundError:
        st.error("Income data file not found. Please ensure the file path is correct.")
        return None


# Main application
def main():
    income_df = load_income_data()
    with st.form(key="input_form"):
        # Inputs inside the form
        st.image('logo.png')
        st.write('Your smart real-estate price predictor Ai-based tool')
        st.write('Fill in the following fields, and we make the magic happen!')
        type_prop = st.selectbox("Type of property", ["House", "Apartment"])
        state_build = st.selectbox(
            "State of building",
            [
                "Good",
                "Just renovated",
                "As New",
                "To be done up",
                "To renovate",
                "To restore",
            ],
        )

        zip_code = st.selectbox(
            "Enter or select a postcode:", options=income_df["zipCode"]
        )
        if zip_code in income_df["zipCode"].values:
            avg_income = income_df.loc[
                income_df["zipCode"] == zip_code, "Average_Income_Per_Citizen"
            ].values[0]

        # Determine province based on zip code
        if 1000 <= zip_code <= 1299:
            province = "Brussel"
        elif 1300 <= zip_code <= 1499:
            province = "Waals-Brabant"
        elif 1500 <= zip_code <= 1999:
            province = "Vlaams-Brabant"
        elif 2000 <= zip_code <= 2999:
            province = "Antwerpen"
        elif 3000 <= zip_code <= 3499:
            province = "Vlaams-Brabant"
        elif 3500 <= zip_code <= 3999:
            province = "Limburg"
        elif 4000 <= zip_code <= 4999:
            province = "Luik"
        elif 5000 <= zip_code <= 5680:
            province = "Namen"
        elif 6000 <= zip_code <= 6599:
            province = "Henegouwen"
        elif 6600 <= zip_code <= 6999:
            province = "Luxemburg"
        elif 7000 <= zip_code <= 7999:
            province = "Henegouwen"
        elif 8000 <= zip_code <= 8999:
            province = "West-Vlaanderen"
        elif 9000 <= zip_code <= 9999:
            province = "Oost-Vlaanderen"
        else:
            province = None
        
        if province:
            st.write(f"Province: {province}")

        livingarea = st.number_input("Living Area (m²)", min_value=0.0, value=100.0)
        bedrooms = st.number_input(
            "Number of Bedrooms", min_value=0, max_value=10, value=3
        )
        kitchen = st.selectbox("Equipped Kitchen?", ["Yes", "No"])
        facades = st.number_input(
            "Number of Facades", min_value=1, max_value=4, value=2
        )

        terrace = st.selectbox("Has terrace?", ["Yes", "No"])
        garden = st.selectbox("Has garden?", ["Yes", "No"])
        if garden == "Yes":
            gardensurface = st.number_input(
                "Garden Surface (m²)", min_value=0.0, value=50.0
            )
        else:
            gardensurface = 0

        # Submit button for the form
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Initialize all features with defaults
        input_data = {
            col: 0
            for col in [
                "bedrooms",
                "kitchen",
                "facades",
                "terrace",
                "gardensurface",
                "livingarea",
                "surfaceoftheplot",
                "as_new",
                "good",
                "just_renovated",
                "to_be_done_up",
                "to_renovate",
                "to_restore",
                "is_apartment",
                "is_house",
                "Average_Income_Per_Citizen",
                "province_Antwerpen",
                "province_Brussel",
                "province_Henegouwen",
                "province_Limburg",
                "province_Luik",
                "province_Luxemburg",
                "province_Namen",
                "province_Oost-Vlaanderen",
                "province_Vlaams-Brabant",
                "province_Waals-Brabant",
                "province_West-Vlaanderen",
            ]
        }

        # Update input data with user inputs
        if province:
            input_data.update(
                {
                    f"province_{province}": 1,
                    "bedrooms": bedrooms,
                    "is_house": 1 if type_prop == "House" else 0,
                    "is_apartment": 1 if type_prop == "Apartment" else 0,
                    "kitchen": 1 if kitchen == "Yes" else 0,
                    "livingarea": livingarea,
                    "terrace": 1 if terrace == "Yes" else 0,
                    "facades": facades,
                    "Average_Income_Per_Citizen": avg_income,
                    "gardensurface": gardensurface,
                    "as_new": 1 if state_build == "As New" else 0,
                    "good": 1 if state_build == "Good" else 0,
                    "just_renovated": 1 if state_build == "Just renovated" else 0,
                    "to_be_done_up": 1 if state_build == "To be done up" else 0,
                    "to_renovate": 1 if state_build == "To renovate" else 0,
                    "to_restore": 1 if state_build == "To restore" else 0,
                }
            )

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the data
        preprocessed_data = preprocess(input_df)

        # Load model
        model = load_model()

        # Predict and display result
        if model:
            try:
                prediction = predict(model, preprocessed_data)
                if prediction is not None:
                    st.write(f"Predicted price based on your informations: {prediction[0]:,.2f}€")
                    st.write(f"We recommend a price between {round(prediction[0] * 0.95 / 1000) * 1000:,}€ and {round(prediction[0] * 1.05 / 1000) * 1000:,}€")


                else:
                    st.error("An error occurred during prediction.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error(
                "Model could not be loaded. Please check the file path and model format."
            )


# Run the app
if __name__ == "__main__":
    main()
