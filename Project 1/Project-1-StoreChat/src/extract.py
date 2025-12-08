import pandas as pd
try:
    from src import logger
except ImportError:
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

#--------------------------------EXTRACT PRODUCTS.CSV------------------------------
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
        "CAT-1": "../dataset/images/Accessories.jpg",
        "CAT-2": "../dataset/images/Audio.jpg",
        "CAT-3": "../dataset/images/Desktop.jpg",
        "CAT-4": "images/Laptop.jpg",
        "CAT-5": "../dataset/images/Smart_Speaker.jpg",
        "CAT-6": "../dataset/images/Smartphone.jpg",
        "CAT-7": "../dataset/images/Streaming_Device.jpg",
        "CAT-8": "../dataset/images/Subscription_Service.jpg",
        "CAT-9": "../dataset/images/Tablet.jpg",
        "CAT-10": "../dataset/images/Wearable.jpg",
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

#--------------------------------EXTRACT SALES.CSV------------------------------
def convert_sales_data_types(df):
    df['sale_id'] = df['sale_id'].astype(str)
    df['sale_date'] = pd.to_datetime(df['sale_date'], format='%d-%m-%Y', errors='coerce')
    df['store_id'] = df['store_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').astype('Int64')
    logger.log_info("EXTRACT: Sales data types converted")
    return df

def add_time_dimensions(df):
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_quarter'] = df['sale_date'].dt.quarter
    df['sale_day_of_week'] = df['sale_date'].dt.day_name()
    df['sale_week'] = df['sale_date'].dt.isocalendar().week
    logger.log_info("EXTRACT: Time dimensions added")
    return df

def validate_sales_data(df):
    validation_results = {
        'missing_sale_id': df['sale_id'].isna().sum(),
        'missing_sale_date': df['sale_date'].isna().sum(),
        'missing_store_id': df['store_id'].isna().sum(),
        'missing_product_id': df['product_id'].isna().sum(),
        'missing_quantity': df['quantity'].isna().sum(),
        'zero_or_negative_quantity': (df['quantity'] <= 0).sum(),
        'invalid_date_format': df['sale_date'].isna().sum(),
        'future_dates': (df['sale_date'] > pd.Timestamp.now()).sum(),
    }
    return validation_results

def deduplicate_sales(df):
    duplicates = df.duplicated(subset=['sale_id'], keep=False)
    duplicate_count = duplicates.sum()
    df_clean = df.drop_duplicates(subset=['sale_id'], keep='first')
    if duplicate_count > 0:
        logger.log_warning(f"EXTRACT: Removed {duplicate_count} duplicate sale_ids")
    return df_clean

def process_sales(input_file, output_file):
    df = pd.read_csv(input_file)
    df = clean_column_names(df)
    df = convert_sales_data_types(df)
    df = add_time_dimensions(df)
    df = deduplicate_sales(df)
    
    validation_results = validate_sales_data(df)
    
    df = df[[
        'sale_id', 'sale_date', 'store_id', 'product_id', 'quantity',
        'sale_year', 'sale_month', 'sale_quarter', 'sale_day_of_week', 'sale_week'
    ]]
    
    df.to_csv(output_file, index=False)
    logger.log_info("EXTRACT: Sales processed successfully")
    return df, validation_results

if __name__ == "__main__": # pragma: no cover
    #--------------------------------Process PRODUCTS.CSV------------------------------
    df = pd.read_csv("../dataset/products.csv")
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
    
    df.to_csv("../dataset/products_with_images.csv", index=False)
    print("Extract: Products processed")

    #--------------------------------Process SALES.CSV------------------------------
    print("\n\n---------------Processing Sales Data-----------------")
    sales_df, sales_validation = process_sales(
        "../dataset/sales.csv",
        "../dataset/sales_processed.csv"
    )
    
    print("\n---------------Sales Validation Checks-----------------")
    for key, value in sales_validation.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\nProcessed {len(sales_df)} sales records")