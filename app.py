import streamlit as st
import pandas as pd
import pickle
from preprocessing.cleaning_data import preprocess  # Import your preprocess function
from predict.prediction import predict

# Load the model
@st.cache_resource
def load_model():
    model_path = 'models/random_forest.pkl'  # Adjust path as needed
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
    income_path = r'cleaned_income_data.csv'  # Ensure the correct path to your CSV file
    try:
        income_df = pd.read_csv(income_path)
        return income_df
    except FileNotFoundError:
        st.error("Income data file not found. Please ensure the file path is correct.")
        return None
    
# Main application
def main():
    income_df = load_income_data()
    st.title("ImmoEliza ML Deployment")
    st.write("Predict property prices using our model.")
    st.sidebar.header("Input Features")
    zip_code = st.sidebar.number_input("Enter Zip Code", min_value=1000, max_value=9999, value=1000)
    # Look up average income based on the entered zip code
    if zip_code in income_df['zipCode'].values:
        avg_income = income_df.loc[income_df['zipCode'] == zip_code, 'Average_Income_Per_Citizen'].values[0]
    bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
    kitchen = st.sidebar.selectbox("Equipped Kitchen?", ["Yes", "No"])
    livingarea = st.sidebar.number_input("Living Area (m²)", min_value=0.0, value=100.0)
    terrace = st.sidebar.selectbox("Has Terrace?", ["Yes", "No"])
    facades = st.sidebar.number_input("Number of Facades", min_value=1, max_value=4, value=2)
    type_prop = st.sidebar.selectbox("Type of property", ["House", "Apartment"])
    province = st.sidebar.selectbox(
        "Province",
        ["Brussel", "Luik", "Namen", "Antwerpen", "Henegouwen", "Limburg", 
         "Luxemburg", "Oost-Vlaanderen", "Vlaams-Brabant", "Waals-Brabant", "West-Vlaanderen"]
    )
    gardensurface = st.sidebar.number_input("Garden Surface (m²)", min_value=0.0, value=50.0)

    # Initialize all features with defaults
    input_data = {col: 0 for col in [
        "bedrooms", "kitchen", "facades", "terrace", "gardensurface", "livingarea", 
        "surfaceoftheplot", "as_new", "good", "just_renovated", "to_be_done_up", 
        "to_renovate", "to_restore", "is_apartment", "is_house", 
        "Average_Income_Per_Citizen", "province_Antwerpen", "province_Brussel", 
        "province_Henegouwen", "province_Limburg", "province_Luik", "province_Luxemburg", 
        "province_Namen", "province_Oost-Vlaanderen", "province_Vlaams-Brabant", 
        "province_Waals-Brabant", "province_West-Vlaanderen"
    ]}
    
    # Update input data with user inputs
    input_data.update({
        "bedrooms": bedrooms,
        "is_house": 1 if type_prop == "House" else 0,
        "is_apartment": 1 if type_prop == "Apartment" else 0,
        "kitchen": 1 if kitchen == "Yes" else 0,
        "livingarea": livingarea,
        "terrace": 1 if terrace == "Yes" else 0,
        "facades": facades,
        "Average_Income_Per_Citizen": avg_income,
        "gardensurface": gardensurface,
        "province_" + province: 1,
    })
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the data
    preprocessed_data = preprocess(input_df)

    # Load model
    model = load_model()

    # Predict and display result
    if model:
        if st.button("Predict"):
            try:
                prediction = predict(model, preprocessed_data)
                if prediction is not None:
                    st.write(f"Predicted Price: €{prediction[0]:,.2f}")
                else:
                    st.error("An error occurred during prediction.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.error("Model could not be loaded. Please check the file path and model format.")


# Run the app
if __name__ == "__main__":
    main()