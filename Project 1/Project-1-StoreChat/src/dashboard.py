import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard_service import DashboardService

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="StoreChat Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- SERVICES --------------------
@st.cache_resource
def get_service():
    """Singleton for DashboardService"""
    return DashboardService()

service = get_service()

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data():
    """Load all base datasets"""
    summary_df = service.load_summary_data()
    return summary_df

summary_df = load_data()

if summary_df.empty:
    st.error("Unable to load data. Please check if the dataset files exist.")
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.header("🔍 Filters")

# Year Filter
available_years = sorted(summary_df['sale_year'].unique())
selected_year = st.sidebar.selectbox(
    "Select Year", 
    available_years, 
    index=len(available_years)-1 if available_years else 0
)

# Store Filter
available_stores = sorted(summary_df['store_id'].unique())
selected_store = st.sidebar.selectbox(
    "Select Store (for specific charts)", 
    available_stores
)

# -------------------- MAIN CONTENT --------------------
st.title("📊 StoreChat Analytics Dashboard")
st.markdown(f"**Year:** {selected_year}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Overview", "💰 Revenue", "📋 Data Tables"])

# -------------------- TAB 1: OVERVIEW --------------------
with tab1:
    col1, col2 = st.columns(2)
    
    # Chart 1: Total Quantity by Store (Filtered by Year)
    with col1:
        st.subheader(f"Total Quantity Sold by Store ({selected_year})")
        
        # Aggregate data for the selected year
        yearly_summary = summary_df[summary_df['sale_year'] == selected_year]
        qty_by_store = yearly_summary.groupby('store_id')['total_quantity'].sum().reset_index()
        
        if not qty_by_store.empty:
            fig_qty = px.bar(
                qty_by_store, 
                x='store_id', 
                y='total_quantity',
                labels={'store_id': 'Store', 'total_quantity': 'Quantity Sold'},
                color='total_quantity',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_qty, use_container_width=True)
        else:
            st.info("No data available for this year.")

    # Chart 2: Monthly Trend for Selected Store
    with col2:
        st.subheader(f"Monthly Sales Trend: {selected_store} ({selected_year})")
        
        store_monthly = summary_df[
            (summary_df['store_id'] == selected_store) & 
            (summary_df['sale_year'] == selected_year)
        ].sort_values('sale_month')
        
        if not store_monthly.empty:
            # Create a more readable month label
            store_monthly['month_name'] = pd.to_datetime(
                store_monthly['sale_month'], format='%m'
            ).dt.month_name()
            
            fig_trend = px.line(
                store_monthly,
                x='month_name',
                y='total_quantity',
                markers=True,
                labels={'month_name': 'Month', 'total_quantity': 'Quantity'},
                title=f"Sales Trend for {selected_store}"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info(f"No monthly data found for {selected_store} in {selected_year}.")

# -------------------- TAB 2: REVENUE --------------------
with tab2:
    st.subheader(f"Revenue Analysis ({selected_year})")
    
    with st.spinner("Calculating revenue..."):
        # Load revenue data (calculated on the fly as it involves joining large tables)
        revenue_df = service.get_revenue_by_store(selected_year)
    
    if not revenue_df.empty:
        col_rev_1, col_rev_2 = st.columns([2, 1])
        
        with col_rev_1:
            fig_rev = px.bar(
                revenue_df,
                x='store_id',
                y='revenue',
                labels={'store_id': 'Store', 'revenue': 'Total Revenue ($)'},
                color='revenue',
                color_continuous_scale=px.colors.sequential.Plasma,
                text_auto='.2s'
            )
            fig_rev.update_layout(xaxis_title="Store", yaxis_title="Revenue ($)")
            st.plotly_chart(fig_rev, use_container_width=True)
            
        with col_rev_2:
            st.dataframe(
                revenue_df.style.format({"revenue": "${:,.2f}"}), 
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning(f"No revenue data available for {selected_year}. Check if sales and products data exist.")

# -------------------- TAB 3: DATA TABLES --------------------
with tab3:
    st.subheader("Raw Data Explorer")
    
    dataset_option = st.selectbox("Select Dataset", ["Store Sales Summary", "Sales Transactions", "Products"])
    
    if dataset_option == "Store Sales Summary":
        st.dataframe(summary_df, use_container_width=True)
        
    elif dataset_option == "Sales Transactions":
        with st.spinner("Loading sales data..."):
            sales_limit = st.slider("Max rows to display", 100, 5000, 1000)
            sales_df = service.load_sales_data().head(sales_limit)
            st.dataframe(sales_df, use_container_width=True)
            
    elif dataset_option == "Products":
        with st.spinner("Loading products..."):
            products_df = service.load_products_data()
            st.dataframe(products_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("StoreChat Dashboard v1.0 | Generated by Assistant")

