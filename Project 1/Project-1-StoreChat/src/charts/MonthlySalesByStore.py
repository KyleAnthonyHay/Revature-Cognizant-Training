import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv('../../dataset/store_sales_summary.csv')

#--------------------------------Monthly Sales Chart--------------------------------
df['date'] = pd.to_datetime(df['sale_year'].astype(str) + '-' + df['sale_month'].astype(str) + '-01')

fig = px.line(df[df['store_id'] == 'ST-1'], 
              x='date', y='total_quantity',
              title='Store ST-1 Monthly Sales')
       
fig.show()