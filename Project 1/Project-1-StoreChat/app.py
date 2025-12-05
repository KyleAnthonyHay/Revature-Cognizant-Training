from flask import Flask, request, jsonify
from repo import init_db, load_data

# app = Flask(__name__)

if __name__ == "__main__":
    init_db()        
    load_data()         # create table if missing
    # app.run(host="127.0.0.1", port=8000, debug=True)


"""
source venv/bin/activate
"""  