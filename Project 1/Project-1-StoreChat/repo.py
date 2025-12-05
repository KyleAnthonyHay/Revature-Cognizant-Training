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

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )

def init_db():
    with get_conn() as conn, conn.cursor() as cur:
        try:
            logger.log_info("Opening schema.sql file")
            with open("schema.sql", "r", encoding="utf-8") as f:
                cur.execute(f.read())
            logger.log_info("Schema.sql file opened successfully")

        except FileNotFoundError:
            logger.log_error("Schema.sql file not found")
            raise 

        except Exception as e:
            logger.log_error(f"Error opening schema.sql file: {e}")
            raise 

        conn.commit()

def load_data():
    """ Load data from CSV files into the database """
    with get_conn() as conn, conn.cursor() as cur: # # TODO : add try catch

        #---------------LOAD CATEGORIES CSV FILE---------------
        try:
            logger.log_info("Reading category.csv file")
            category_df = pd.read_csv("dataset/category.csv")
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
            product_df = pd.read_csv("dataset/products_with_images.csv")
            logger.log_info(f"products_with_images.csv file read successfully with {len(product_df)} rows")
        except FileNotFoundError:
            logger.log_error("products_with_images.csv file not found")
            raise
        except Exception as e:
            logger.log_error(f"Error reading products_with_images.csv file: {e}")
            raise
        #---------------VALIDATE PRODUCTS DATAFRAME---------------
        product_df, rejected_df = validateService.clean_dataframe(product_df, True)
        #---------------SAVE REJECTED PRODUCTS---------------
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