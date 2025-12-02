import os
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
import validateService

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
        with open("schema.sql", "r", encoding="utf-8") as f: # TODO : add try catch
            cur.execute(f.read())
        conn.commit()

def load_data():
    """ Load data from CSV files into the database """
    with get_conn() as conn, conn.cursor() as cur: # # TODO : add try catch
        # Load categories first (required for foreign key)
        category_df = pd.read_csv("dataset/category.csv") # TODO : add try catch
        for _, row in category_df.iterrows():
            cur.execute("INSERT INTO categories (category_id, category_name) VALUES (%s, %s) ON CONFLICT (category_id) DO NOTHING",
            (row["category_id"], row["category_name"])
            )

        # Load and validate Products
        product_df = pd.read_csv("dataset/products_with_images.csv")
        product_df, rejected_df = validateService.clean_dataframe(product_df, True)
        print(f"Rejected {len(rejected_df)} rows")
        
        for _, row in product_df.iterrows():
            cur.execute("INSERT INTO products (product_id, product_name, category_id, launch_date, price, image_url) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (product_id) DO NOTHING",
            (row['product_id'], row['product_name'], row['category_id'], 
                 row['launch_date'], row['price'], row['image_url'])
            )

        conn.commit()
        print(f"Loaded {len(category_df)} categories and {len(product_df)} products")