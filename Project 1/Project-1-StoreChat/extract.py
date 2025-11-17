import pandas as pd

df  = pd.read_csv("dataset/products.csv")
print(df.head())

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

#Convert Prices to floats
df['price'] = df['price'].astype(float)
df['launch_date'] = pd.to_datetime(df['launch_date'])
df["product_name"] = df["product_name"].astype(str)

print("---------------DTypes After Conversion-----------------")
print(df.dtypes)

# Step 4: Basic Validation
print("\n---------------Validation Checks-----------------")
missing_product_id = df['product_id'].isna().sum()
missing_name = df['product_name'].isna().sum()
missing_category_id = df['category_id'].isna().sum()
missing_launch_date = df['launch_date'].isna().sum()
missing_price = df['price'].isna().sum()
negative_price = (df['price'] <0).sum()
invalid_launch_date = df['launch_date'].isna().sum()

print(f"Missing Product ID: {missing_product_id}")
print(f"Missing Name: {missing_name}")
print(f"Missing Category ID: {missing_category_id}")
print(f"Missing Launch Date: {missing_launch_date}")
print(f"Missing Price: {missing_price}")
print(f"Negative Price: {negative_price}")
print(f"Invalid Launch Date: {invalid_launch_date}")

print("\n---------------Add Category Images-----------------")
# create dictionary to map category_id to image_url
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

df['image_url'] = (
    df['category_id'].map(category_mapping)
)

df.to_csv("dataset/products_with_images.csv", index=False)
print(df) 
