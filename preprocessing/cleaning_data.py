import pandas as pd


def preprocess(data):
    # Convert the input data to a DataFrame
    df = pd.DataFrame(data)

    # Define required columns
    required_columns = [
        "bedrooms",
        "kitchen",
        "facades",
        "terrace",
        "livingarea",
        "Average_Income_Per_Citizen",
        "surfaceoftheplot",
        "as_new",
        "province_Brussel",
        "province_Luik",
        "province_Namen",
        "good",
        "to_renovate",
        "is_house",
        "is_apartment",
        "gardensurface",
        "just_renovated",
        "to_be_done_up",
        "to_restore",
        "province_Antwerpen",
        "province_Henegouwen",
        "province_Limburg",
        "province_Luxemburg",
        "province_Oost-Vlaanderen",
        "province_Vlaams-Brabant",
        "province_Waals-Brabant",
        "province_West-Vlaanderen",
    ]

    # Add missing columns with default values (e.g., 0)
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default value, modify if needed

    # Fill missing values (NaN) for numerical columns with 0
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(0)

    # Return the preprocessed data
    return df
