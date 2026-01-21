import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

print("Connecting to Postgres...")

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode=os.getenv("PGSSLMODE", "require"),
)

cur = conn.cursor()

cur.execute("""
    SELECT
        current_user,
        current_database(),
        inet_server_addr(),
        now();
""")

print("Connection OK:")
print(cur.fetchone())

cur.close()
conn.close()
