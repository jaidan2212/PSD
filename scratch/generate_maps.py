import pandas as pd
import folium
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset_sentinel2_jatim.csv')

FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
X = df[FITUR]
y = df['kelas_id']
kelas_map = dict(zip(df['kelas_id'], df['kelas']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
df['prediksi'] = rf.predict(X)

colors = {1: "orange", 2: "red", 3: "purple", 4: "green", 5: "darkblue", 6: "lightblue"}

# Peta Semua Kelas
m = folium.Map(location=[-7.7, 112.5], zoom_start=8)
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google', name='Google Satellite', overlay=False, control=True
).add_to(m)

for idx, row in df.iterrows():
    k_id = row['prediksi']
    n_kelas = kelas_map[k_id]
    folium.CircleMarker(
        location=[row['lat'], row['lon']], radius=5, popup=f"Prediksi: {n_kelas}",
        color=colors.get(k_id, "black"), fill=True, fill_color=colors.get(k_id, "black"), fill_opacity=0.8
    ).add_to(m)

os.makedirs('materi/output', exist_ok=True)
m.save('materi/output/peta_hasil_rf.html')

# Peta Khusus Laut
m_laut = folium.Map(location=[-7.7, 112.5], zoom_start=8)
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google', name='Google Satellite', overlay=False, control=True
).add_to(m_laut)

df_laut = df[df['prediksi'] == 5]
for idx, row in df_laut.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']], radius=6, popup=f"Titik Laut (lon: {row['lon']:.4f}, lat: {row['lat']:.4f})",
        color="cyan", fill=True, fill_color="blue", fill_opacity=0.8
    ).add_to(m_laut)

m_laut.save('materi/output/peta_khusus_laut.html')
