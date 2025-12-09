import os
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
import validateService
import logger

DB_NAME = os.getenv("PGDB", "storechat")
DB_USER = os.getenv("PGUSER", "")
DB_PASS = os.getenv("PGPASS", "") 
DB_HOST = os.getenv("PGHOST", "127.0.0.1")
DB_PORT = int(os.getenv("PGPORT", "5432"))


#--------------------------------CONNECTION FUNCTIONS------------------------------
def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )

def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        try:
            logger.log_info("Opening schema.sql file")
            with open("../schema.sql", "r", encoding="utf-8") as f:
                cur.execute(f.read())
            logger.log_info("Schema.sql file opened successfully")

        except FileNotFoundError:
            logger.log_error("Schema.sql file not found")
            raise 

        except Exception as e:
            logger.log_error(f"Error opening schema.sql file: {e}")
            raise 

        conn.commit()

#--------------------------------LOAD STORES CSV FILE------------------------------
def load_stores():
    """Load stores data from CSV file into the database"""
    with get_conn() as conn, conn.cursor() as cur:
        try:
            logger.log_info("Reading stores.csv file")
            stores_df = pd.read_csv("../dataset/stores.csv")
            stores_df.columns = stores_df.columns.str.strip().str.lower().str.replace(" ", "_")
            logger.log_info(f"stores.csv file read successfully with {len(stores_df)} rows")
        except FileNotFoundError:
            logger.log_error("stores.csv file not found")
            raise
        except Exception as e:
            logger.log_error(f"Error reading stores.csv file: {e}")
            raise
        
        stores_df, rejected_df = validateService.clean_dataframe(stores_df, False)
        
        if len(rejected_df) > 0:
            for _, row in rejected_df.iterrows():
                reason = row.get('rejection_reason', 'Validation failed')
                row_data = row.drop('rejection_reason').to_dict() if 'rejection_reason' in row else row.to_dict()
                cur.execute(
                    "INSERT INTO rejected_fields (source_table, rejection_reason, rejected_data) VALUES (%s, %s, %s)",
                    ("stores", reason, pd.Series(row_data).to_json())
                )
        print(f"Rejected {len(rejected_df)} rows from stores.csv file")
        
        for _, row in stores_df.iterrows():
            cur.execute(
                "INSERT INTO stores (store_id, store_name, city, country) VALUES (%s, %s, %s, %s) ON CONFLICT (store_id) DO NOTHING",
                (row["store_id"], row["store_name"], row["city"], row["country"])
            )
        
        conn.commit()
        print(f"Loaded {len(stores_df)} stores")

def load_sales():
    """Load sales data from CSV file into the database"""
    with get_conn() as conn, conn.cursor() as cur:
        try:
            logger.log_info("Reading sales_processed.csv file")
            sales_df = pd.read_csv("../dataset/sales_processed.csv")
            logger.log_info(f"sales_processed.csv file read successfully with {len(sales_df)} rows")
        except FileNotFoundError:
            logger.log_error("sales_processed.csv file not found")
            raise
        except Exception as e:
            logger.log_error(f"Error reading sales_processed.csv file: {e}")
            raise
        
        sales_df, rejected_df = validateService.clean_dataframe(sales_df, False)
        
        if len(rejected_df) > 0:
            for _, row in rejected_df.iterrows():
                reason = row.get('rejection_reason', 'Validation failed')
                row_data = row.drop('rejection_reason').to_dict() if 'rejection_reason' in row else row.to_dict()
                cur.execute(
                    "INSERT INTO rejected_fields (source_table, rejection_reason, rejected_data) VALUES (%s, %s, %s)",
                    ("sales", reason, pd.Series(row_data).to_json())
                )
        print(f"Rejected {len(rejected_df)} rows from sales_processed.csv file")
        
        batch_size = 1000
        total_rows = len(sales_df)
        
        for i in range(0, total_rows, batch_size):
            batch = sales_df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO sales (
                            sale_id, sale_date, store_id, product_id, quantity,
                            sale_year, sale_month, sale_quarter, sale_day_of_week, sale_week
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sale_id) DO NOTHING
                    """, (
                        row['sale_id'], row['sale_date'], row['store_id'], row['product_id'], int(row['quantity']),
                        int(row['sale_year']), int(row['sale_month']), int(row['sale_quarter']), 
                        row['sale_day_of_week'], int(row['sale_week'])
                    ))
                except Exception as e:
                    logger.log_error(f"Error inserting sale {row['sale_id']}: {e}")
                    continue
            
            if (i + batch_size) % 10000 == 0:
                logger.log_info(f"Loaded {min(i + batch_size, total_rows)} of {total_rows} sales records")
        
        conn.commit()
        print(f"Loaded {len(sales_df)} sales")

def load_store_sales_summary():
    """Load store sales summary data from CSV file into the database"""
    with get_conn() as conn, conn.cursor() as cur:
        try:
            logger.log_info("Reading store_sales_summary.csv file")
            summary_df = pd.read_csv("../dataset/store_sales_summary.csv")
            logger.log_info(f"store_sales_summary.csv file read successfully with {len(summary_df)} rows")
        except FileNotFoundError:
            logger.log_error("store_sales_summary.csv file not found")
            raise
        except Exception as e:
            logger.log_error(f"Error reading store_sales_summary.csv file: {e}")
            raise
        
        summary_df, rejected_df = validateService.clean_dataframe(summary_df, False)
        
        if len(rejected_df) > 0:
            for _, row in rejected_df.iterrows():
                reason = row.get('rejection_reason', 'Validation failed')
                row_data = row.drop('rejection_reason').to_dict() if 'rejection_reason' in row else row.to_dict()
                cur.execute(
                    "INSERT INTO rejected_fields (source_table, rejection_reason, rejected_data) VALUES (%s, %s, %s)",
                    ("store_sales_summary", reason, pd.Series(row_data).to_json())
                )
        logger.log_info(f"Rejected {len(rejected_df)} rows from store_sales_summary.csv file")
        
        for _, row in summary_df.iterrows():
            try:
                cur.execute("""
                    INSERT INTO store_sales_summary (
                        store_id, sale_year, sale_month, total_quantity, 
                        total_transactions, avg_quantity_per_transaction
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (store_id, sale_year, sale_month) DO UPDATE SET
                        total_quantity = EXCLUDED.total_quantity,
                        total_transactions = EXCLUDED.total_transactions,
                        avg_quantity_per_transaction = EXCLUDED.avg_quantity_per_transaction
                """, (
                    row['store_id'], int(row['sale_year']), int(row['sale_month']),
                    int(row['total_quantity']), int(row['total_transactions']),
                    float(row['avg_quantity_per_transaction'])
                ))
            except Exception as e:
                logger.log_error(f"Error inserting store sales summary for {row['store_id']}: {e}")
                continue
        
        conn.commit()
        logger.log_info(f"Loaded {len(summary_df)} store sales summary records")
        print(f"Loaded {len(summary_df)} store sales summary records")

def load_data():
    """ Load data from CSV files into the database """
    with get_conn() as conn, conn.cursor() as cur: # # TODO : add try catch

        #---------------LOAD CATEGORIES CSV FILE---------------
        try:
            logger.log_info("Reading category.csv file")
            category_df = pd.read_csv("../dataset/category.csv")
            logger.log_info(f"Category.csv file read successfully with {len(category_df)} rows")
        except FileNotFoundError:
            logger.log_error("Category.csv file not found")
            raise
        except Exception as e:
            logger.log_error(f"Error reading category.csv file: {e}")
            raise
        #---------------VALIDATE CATEGORIES DATAFRAME---------------
        category_df, rejected_df = validateService.clean_dataframe(category_df, False)
        #---------------SAVE REJECTED CATEGORIES---------------
        if len(rejected_df) > 0:
            for _, row in rejected_df.iterrows():
                reason = row.get('rejection_reason', 'Validation failed')
                row_data = row.drop('rejection_reason').to_dict() if 'rejection_reason' in row else row.to_dict()
                cur.execute(
                    "INSERT INTO rejected_fields (source_table, rejection_reason, rejected_data) VALUES (%s, %s, %s)",
                    ("categories", reason, pd.Series(row_data).to_json())
                )
        print(f"Rejected {len(rejected_df)} rows from category.csv file")
        #---------------LOAD CATEGORIES INTO DATABASE---------------
        for _, row in category_df.iterrows():
            cur.execute("INSERT INTO categories (category_id, category_name) VALUES (%s, %s) ON CONFLICT (category_id) DO NOTHING",
            (row["category_id"], row["category_name"])
            )

        #---------------LOAD PRODUCTS CSV FILE---------------
        try:
            logger.log_info("Reading products_with_images.csv file")
            product_df = pd.read_csv("../dataset/products_with_images.csv")
            logger.log_info(f"products_with_images.csv file read successfully with {len(product_df)} rows")
        except FileNotFoundError:
            logger.log_error("products_with_images.csv file not found")
            raise
        except Exception as e:
            logger.log_error(f"Error reading products_with_images.csv file: {e}")
            raise
        #---------------VALIDATE PRODUCTS DATAFRAME---------------
        product_df, rejected_df = validateService.clean_dataframe(product_df, True)
        #---------------LOAD REJECTED PRODUCTS INTO DATABASE---------------
        if len(rejected_df) > 0:
            for _, row in rejected_df.iterrows():
                reason = row.get('rejection_reason', 'Validation failed')
                row_data = row.drop('rejection_reason').to_dict() if 'rejection_reason' in row else row.to_dict()
                cur.execute(
                    "INSERT INTO rejected_fields (source_table, rejection_reason, rejected_data) VALUES (%s, %s, %s)",
                    ("products", reason, pd.Series(row_data).to_json())
                )
        print(f"Rejected {len(rejected_df)} rows from products_with_images.csv file")
        #---------------LOAD PRODUCTS INTO DATABASE---------------
        for _, row in product_df.iterrows():
            cur.execute("INSERT INTO products (product_id, product_name, category_id, launch_date, price, image_url) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (product_id) DO NOTHING",
            (row['product_id'], row['product_name'], row['category_id'], 
                 row['launch_date'], row['price'], row['image_url'])
            )

        conn.commit()
        print(f"Loaded {len(category_df)} categories and {len(product_df)} products")
        load_stores()
        load_sales()
        load_store_sales_summary()