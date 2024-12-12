import pandas as pd
import logging
def predict(model, X):
    model_feature_names = [
    "bedrooms", "kitchen", "facades", "terrace", "gardensurface", "livingarea", 
    "surfaceoftheplot", "as_new", "good", "just_renovated", "to_be_done_up", 
    "to_renovate", "to_restore", "is_apartment", "is_house", 
    "Average_Income_Per_Citizen", "province_Antwerpen", "province_Brussel", 
    "province_Henegouwen", "province_Limburg", "province_Luik", "province_Luxemburg", 
    "province_Namen", "province_Oost-Vlaanderen", "province_Vlaams-Brabant", 
    "province_Waals-Brabant", "province_West-Vlaanderen"
]
    if model:
        try:
            X = X[model_feature_names]
            predictions = model.predict(X)
            logging.info(f"Predictions: {predictions}")
            return predictions
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return None
    return None


import pickle

model_path = 'models/random_forest.pkl'
with open(model_path, "rb") as file:
    model = pickle.load(file)
print(type(model))
