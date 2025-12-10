import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv('../../dataset/store_sales_summary.csv')

#--------------------------------Calculate Revenue--------------------------------
sales = pd.read_csv('../../dataset/sales.csv')
products = pd.read_csv('../../dataset/products.csv')

# Parse dates and filter 2023
sales['sale_date'] = pd.to_datetime(sales['sale_date'], format='%d-%m-%Y')
sales_2023 = sales[sales['sale_date'].dt.year == 2023]

# Join with products to get price
merged = sales_2023.merge(products, left_on='product_id', right_on='Product_ID')
merged['revenue'] = merged['quantity'] * merged['Price']

# Sum by store
revenue_by_store = merged.groupby('store_id')['revenue'].sum().reset_index()

fig = px.bar(revenue_by_store, x='store_id', y='revenue',
             title='Revenue by Store (2023)')
             
fig.show()