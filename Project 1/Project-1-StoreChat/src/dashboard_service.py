import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardService:
    """
    Facade for retrieving and processing dashboard data.
    """
    
    def __init__(self, dataset_path: str = "../dataset"):
        """
        Initialize the service with the path to the dataset directory.
        Resolves the path relative to the current working directory.
        """
        # Try to resolve the path relative to the file location if possible, 
        # or rely on the passed path relative to CWD.
        # Assuming the app is run from 'src/' usually.
        self.dataset_path = Path(dataset_path)
        
    def load_summary_data(self) -> pd.DataFrame:
        """
        Load store sales summary data.
        """
        try:
            file_path = self.dataset_path / "store_sales_summary.csv"
            df = pd.read_csv(file_path)
            # Ensure proper types
            df['sale_year'] = df['sale_year'].astype(int)
            df['sale_month'] = df['sale_month'].astype(int)
            # Create a date column for plotting
            df['date'] = pd.to_datetime(
                df['sale_year'].astype(str) + '-' + df['sale_month'].astype(str) + '-01'
            )
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading summary data: {e}")
            return pd.DataFrame()

    def load_sales_data(self) -> pd.DataFrame:
        """
        Load sales transaction data.
        """
        try:
            file_path = self.dataset_path / "sales.csv"
            df = pd.read_csv(file_path)
            df['sale_date'] = pd.to_datetime(df['sale_date'], format='%d-%m-%Y', errors='coerce')
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading sales data: {e}")
            return pd.DataFrame()

    def load_products_data(self) -> pd.DataFrame:
        """
        Load product information.
        """
        try:
            file_path = self.dataset_path / "products.csv"
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading products data: {e}")
            return pd.DataFrame()

    def get_revenue_by_store(self, year: int) -> pd.DataFrame:
        """
        Calculate revenue by store for a specific year.
        Merges sales and products data.
        """
        sales = self.load_sales_data()
        products = self.load_products_data()
        
        if sales.empty or products.empty:
            return pd.DataFrame(columns=['store_id', 'revenue'])

        # Filter by year
        sales_filtered = sales[sales['sale_date'].dt.year == year]
        
        if sales_filtered.empty:
            return pd.DataFrame(columns=['store_id', 'revenue'])

        # Merge
        merged = sales_filtered.merge(
            products, 
            left_on='product_id', 
            right_on='Product_ID', 
            how='inner'
        )
        
        # Calculate revenue
        merged['revenue'] = merged['quantity'] * merged['Price']
        
        # Group by store
        revenue_by_store = merged.groupby('store_id')['revenue'].sum().reset_index()
        return revenue_by_store.sort_values('revenue', ascending=False)

