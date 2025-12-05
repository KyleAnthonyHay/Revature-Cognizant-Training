import pandas as pd
import logger

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    logger.log_info("EXRACT: Column names cleaned")
    return df

def convert_data_types(df):
    df['price'] = df['price'].astype(float)
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    df["product_name"] = df["product_name"].astype(str)
    logger.log_info("EXRACT: Data types converted")
    return df

def validate_data(df):
    validation_results = {
        'missing_product_id': df['product_id'].isna().sum(),
        'missing_name': df['product_name'].isna().sum(),
        'missing_category_id': df['category_id'].isna().sum(),
        'missing_launch_date': df['launch_date'].isna().sum(),
        'missing_price': df['price'].isna().sum(),
        'negative_price': (df['price'] < 0).sum(),
        'invalid_launch_date': df['launch_date'].isna().sum(),
    }
    return validation_results

def add_category_images(df):
    category_mapping = {
        "CAT-1": "images/Accessories.jpg",
        "CAT-2": "images/Audio.jpg",
        "CAT-3": "images/Desktop.jpg",
        "CAT-4": "images/Laptop.jpg",
        "CAT-5": "images/Smart_Speaker.jpg",
        "CAT-6": "images/Smartphone.jpg",
        "CAT-7": "images/Streaming_Device.jpg",
        "CAT-8": "images/Subscription_Service.jpg",
        "CAT-9": "images/Tablet.jpg",
        "CAT-10": "images/Wearable.jpg",
    }
    df['image_url'] = df['category_id'].map(category_mapping)
    logger.log_info("EXRACT: Category images added")
    return df

def process_products(input_file, output_file):
    df = pd.read_csv(input_file)
    df = clean_column_names(df)
    df = convert_data_types(df)
    validation_results = validate_data(df)
    df = add_category_images(df)
    df.to_csv(output_file, index=False)
    logger.log_info("EXRACT: Products processed successfully")
    return df, validation_results

if __name__ == "__main__": # pragma: no cover
    df = pd.read_csv("dataset/products.csv")
    print(df.head())
    
    df = clean_column_names(df)
    df = convert_data_types(df)
    
    print("---------------DTypes After Conversion-----------------")
    print(df.dtypes)
    
    print("\n---------------Validation Checks-----------------")
    validation_results = validate_data(df)
    for key, value in validation_results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n---------------Add Category Images-----------------")
    df = add_category_images(df)
    
    df.to_csv("dataset/products_with_images.csv", index=False)
    print(df)