import pandas as pd
from sqlalchemy import create_engine, text

# 1. Pastikan nama file CSV sesuai
csv_file = "data_polutan_no2.csv"

# 2. PERHATIKAN DI SINI: Harus ada tulisan '+pymysql' setelah kata 'mysql'
DATABASE_URL = "mysql+pymysql://avnadmin:YOUR_PASSWORD_HERE@mysql-1cd4be9b-zaidannabil2212-b12.j.aivencloud.com:25049/defaultdb"

# 3. Membaca data dari CSV
df = pd.read_csv(csv_file)
print("Jumlah data:", len(df))

# 4. Membuat mesin penghubung MySQL dengan SSL (ca.pem)
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "ssl": {
            "ca": "ca.pem" 
        }
    }
)

# 5. Membuat tabel dengan Primary Key lalu memasukkan data
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS kualitas_udara_no2"))
    conn.execute(text("""
        CREATE TABLE kualitas_udara_no2 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Tanggal VARCHAR(255),
            NO2 FLOAT
        )
    """))
    conn.commit()

# Masukkan data ke MySQL
df.to_sql(
    "kualitas_udara_no2", 
    engine,
    if_exists="append",
    index=False
)

print("✅ Data berhasil masuk ke MySQL Aiven!")