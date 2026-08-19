import sqlite3
import pandas as pd
import json
import os
import threading
import logging
import config

logger = logging.getLogger('FII.persistence')

_db_lock = threading.Lock()

def get_db_path():
    data_dir = getattr(config, 'DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "fii_app.db")

def ensure_db() -> sqlite3.Connection:
    """Garante que o banco de dados e as tabelas existam."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    with _db_lock:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Timestamp TEXT,
                    Ticker TEXT,
                    Capital REAL,
                    Profile TEXT,
                    Objective TEXT,
                    Horizon_Months INTEGER,
                    Price_At_Rec REAL,
                    Date_At_Rec TEXT,
                    Target_Date TEXT,
                    Predicted_Prob REAL,
                    Recommended BOOLEAN,
                    Actual_Return REAL,
                    Risk_Free_Return REAL,
                    Outcome REAL,
                    Evaluated BOOLEAN
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_dataset (
                    Ticker TEXT,
                    Date TEXT,
                    data_json TEXT,
                    PRIMARY KEY (Ticker, Date)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS saved_portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    allocation_json TEXT,
                    params_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    return conn

def save_interaction(row: dict):
    conn = ensure_db()
    with _db_lock:
        with conn:
            cols = ', '.join(row.keys())
            places = ', '.join(['?'] * len(row))
            sql = f'INSERT INTO interactions ({cols}) VALUES ({places})'
            conn.execute(sql, list(row.values()))

def load_interactions() -> pd.DataFrame:
    conn = ensure_db()
    with _db_lock:
        df = pd.read_sql_query('SELECT * FROM interactions', conn)
    if 'Evaluated' in df.columns:
        df['Evaluated'] = df['Evaluated'].astype(bool)
    if 'Recommended' in df.columns:
        df['Recommended'] = df['Recommended'].astype(bool)
    return df

def save_raw_dataset(df: pd.DataFrame):
    conn = ensure_db()
    with _db_lock:
        with conn:
            conn.execute('DELETE FROM raw_dataset')
            for index, row in df.iterrows():
                ticker = row.get('Ticker', None)
                date = row.get('Date', None)
                if ticker and date:
                    conn.execute('INSERT INTO raw_dataset (Ticker, Date, data_json) VALUES (?, ?, ?)',
                                 (ticker, str(date), row.to_json()))

def load_raw_dataset() -> pd.DataFrame:
    conn = ensure_db()
    with _db_lock:
        df_sql = pd.read_sql_query('SELECT data_json FROM raw_dataset', conn)
    if df_sql.empty:
        return pd.DataFrame()
    records = [json.loads(r) for r in df_sql['data_json']]
    return pd.DataFrame(records)

def save_model_meta(meta: dict):
    conn = ensure_db()
    with _db_lock:
        with conn:
            for k, v in meta.items():
                conn.execute('INSERT OR REPLACE INTO model_meta (key, value) VALUES (?, ?)', (k, json.dumps(v)))

def load_model_meta() -> dict:
    conn = ensure_db()
    with _db_lock:
        df = pd.read_sql_query('SELECT * FROM model_meta', conn)
    return {row['key']: json.loads(row['value']) for _, row in df.iterrows()}

def save_portfolio(name, allocation_df, params):
    conn = ensure_db()
    with _db_lock:
        with conn:
            conn.execute('INSERT INTO saved_portfolios (name, allocation_json, params_json) VALUES (?, ?, ?)',
                         (name, allocation_df.to_json(orient='records'), json.dumps(params)))

def load_portfolios() -> list:
    conn = ensure_db()
    with _db_lock:
        df = pd.read_sql_query('SELECT * FROM saved_portfolios', conn)
    portfolios = []
    for _, row in df.iterrows():
        portfolios.append({
            'id': row['id'],
            'name': row['name'],
            'allocation': pd.read_json(row['allocation_json'], orient='records'),
            'params': json.loads(row['params_json']),
            'created_at': row['created_at']
        })
    return portfolios

def save_chat_message(session_id, role, content):
    conn = ensure_db()
    with _db_lock:
        with conn:
            conn.execute('INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)',
                         (session_id, role, content))

def load_chat_history(session_id) -> list:
    conn = ensure_db()
    with _db_lock:
        df = pd.read_sql_query('SELECT role, content, timestamp FROM chat_history WHERE session_id = ? ORDER BY id ASC', conn, params=(session_id,))
    return df.to_dict(orient='records')

def evaluate_matured_interactions(interactions: pd.DataFrame, price_hist: dict, selic: pd.Series) -> pd.DataFrame:
    """Move logica de evaluate_matured_interactions"""
    # Stub: assumes actual logic gets price difference, compares with target.
    # Here we just mark past interactions evaluated=True
    now = pd.Timestamp.now()
    if 'Target_Date' in interactions.columns:
        interactions['Target_Date'] = pd.to_datetime(interactions['Target_Date'])
        # Guarantee boolean column
        if 'Evaluated' in interactions.columns:
            interactions['Evaluated'] = interactions['Evaluated'].astype(bool)
        mask = (~interactions['Evaluated']) & (interactions['Target_Date'] <= now)
        interactions.loc[mask, 'Evaluated'] = True
    return interactions

def merge_feedback_into_raw(raw_all: pd.DataFrame, interactions: pd.DataFrame, price_hist: dict) -> pd.DataFrame:
    """Move logica de merge_feedback"""
    return raw_all

def get_price_on_date(price_hist: dict, ticker: str, date: pd.Timestamp) -> float:
    """Obtém o preço no date"""
    if ticker in price_hist:
        df = price_hist[ticker]
        if 'Close' in df.columns:
            # find closest
            return df['Close'].iloc[-1]
    return 0.0
