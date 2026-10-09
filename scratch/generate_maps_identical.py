import pandas as pd
import folium
import os
import math
from folium.raster_layers import ImageOverlay
from folium import FeatureGroup, LayerControl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Setup Data & Model (hanya butuh CSV, tidak butuh NPZ)
df = pd.read_csv('dataset_sentinel2_jatim.csv')
FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
X = df[FITUR]
y = df['kelas_id']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
df['prediksi'] = rf.predict(X)
df['kelas_prediksi'] = df['prediksi'].map(dict(zip(df['kelas_id'], df['kelas'])))

NAMA_KELAS = ["Sawah", "Bangunan", "Mangrove", "Lahan Hijau", "Perairan Terbuka (Laut)", "Danau"]
WARNA_KELAS = {
    "Sawah": "gold",
    "Bangunan": "red",
    "Mangrove": "purple",
    "Lahan Hijau": "green",
    "Perairan Terbuka (Laut)": "darkblue",
    "Danau": "deepskyblue"
}

AOI = [111.0, -8.8, 114.5, -6.5] # Jatim Bounding Box kasar (Lon Min, Lat Max, Lon Max, Lat Min)
pusat = [-7.7, 112.5] # Jatim tengah

peta = folium.Map(location=pusat, zoom_start=8, tiles=None, control_scale=True)

# Basemap
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri", name="Esri Satellite (online)", max_zoom=19, overlay=False, control=True
).add_to(peta)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google Hybrid", name="Google Hybrid (dengan label)", max_zoom=20, overlay=False, control=True, show=False
).add_to(peta)

# Titik sampel per kelas
for nama in NAMA_KELAS:
    grup = FeatureGroup(name=f"Sampel: {nama}", show=True) # Dibuat True agar langsung kelihatan seperti di gambar
    for _, r in df[df["kelas_prediksi"] == nama].iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=4, color="white", weight=1,
            fill=True, fill_color=WARNA_KELAS[nama], fill_opacity=1,
            popup=f"{nama} (lon {r.lon:.4f}, lat {r.lat:.4f})"
        ).add_to(grup)
    grup.add_to(peta)

# Kotak batas AOI
grup_aoi = FeatureGroup(name="Batas area studi")
folium.Rectangle([[AOI[1], AOI[0]], [AOI[3], AOI[2]]], color="white", weight=1.5, fill=False, dash_array="6").add_to(grup_aoi)
grup_aoi.add_to(peta)

# Legenda
item = "".join(
    f'<div style="margin:2px 0"><span style="display:inline-block;width:14px;height:14px;'
    f'background:{WARNA_KELAS[k]};border:1px solid #333;margin-right:6px;vertical-align:middle"></span>{k}</div>'
    for k in NAMA_KELAS)
legenda = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
           f'padding:10px 12px;border:1px solid #888;border-radius:6px;font:13px Arial;">'
           f'<b>Legenda LULC</b><br>{item}</div>')
peta.get_root().html.add_child(folium.Element(legenda))

LayerControl(collapsed=False).add_to(peta)
peta.fit_bounds([[AOI[1], AOI[0]], [AOI[3], AOI[2]]])

os.makedirs('materi/_static', exist_ok=True)
peta.save('materi/_static/peta_hasil_rf.html')

# -- Peta Khusus Laut (Hanya laut saja yang nyala, sisanya mati) --
peta_laut = folium.Map(location=pusat, zoom_start=8, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri", name="Esri Satellite (online)", max_zoom=19, overlay=False, control=True
).add_to(peta_laut)

for nama in NAMA_KELAS:
    is_laut = (nama == "Perairan Terbuka (Laut)")
    grup = FeatureGroup(name=f"Sampel: {nama}", show=is_laut) # Nyalakan hanya Laut
    for _, r in df[df["kelas_prediksi"] == nama].iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=4, color="white", weight=1,
            fill=True, fill_color=WARNA_KELAS[nama], fill_opacity=1,
            popup=f"{nama} (lon {r.lon:.4f}, lat {r.lat:.4f})"
        ).add_to(grup)
    grup.add_to(peta_laut)

LayerControl(collapsed=False).add_to(peta_laut)
peta_laut.save('materi/_static/peta_khusus_laut.html')
