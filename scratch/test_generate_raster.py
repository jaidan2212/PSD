import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
import folium
from folium.raster_layers import ImageOverlay
from folium import LayerControl, FeatureGroup
import base64
import io
import os

print("Starting raster generation...")

# 1. Load data
df = pd.read_csv("dataset_sentinel2_jatim.csv")
FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
         'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
BAND_ASLI = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

X = df[FITUR]
y = df['kelas_id']

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
print("Random Forest model trained successfully.")

# 2. Setup Spatial Grid for East Java
# Bounding box AOI
lat_min, lat_max = -8.8, -6.6
lon_min, lon_max = 111.0, 114.5
H, W = 320, 500

lats = np.linspace(lat_max, lat_min, H)
lons = np.linspace(lon_min, lon_max, W)
lon_grid, lat_grid = np.meshgrid(lons, lats)
grid_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

# 3. Spatial Interpolation of 10 Bands using KNN from real 520 samples
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(df[['lon', 'lat']], df[BAND_ASLI])

grid_bands = knn.predict(grid_coords)
# Add realistic small spatial texture variation
np.random.seed(42)
grid_bands += np.random.normal(0, 0.006, grid_bands.shape)
grid_bands = np.clip(grid_bands, 0.001, 0.999)

df_grid = pd.DataFrame(grid_bands, columns=BAND_ASLI)

# 4. Compute 8 Spectral Indices using exact formulas
e = 1e-9
nd = lambda a, b: (a - b) / (a + b + e)
df_grid["NDVI"]  = nd(df_grid["B8"], df_grid["B4"])
df_grid["NDWI"]  = nd(df_grid["B3"], df_grid["B8"])
df_grid["MNDWI"] = nd(df_grid["B3"], df_grid["B11"])
df_grid["NDBI"]  = nd(df_grid["B11"], df_grid["B8"])
df_grid["NDRE"]  = nd(df_grid["B8"], df_grid["B5"])
df_grid["EVI"]   = 2.5 * (df_grid["B8"] - df_grid["B4"]) / (df_grid["B8"] + 6 * df_grid["B4"] - 7.5 * df_grid["B2"] + 1)
df_grid["SAVI"]  = 1.5 * (df_grid["B8"] - df_grid["B4"]) / (df_grid["B8"] + df_grid["B4"] + 0.5)
df_grid["BSI"]   = (((df_grid["B11"] + df_grid["B4"]) - (df_grid["B8"] + df_grid["B2"])) /
                    ((df_grid["B11"] + df_grid["B4"]) + (df_grid["B8"] + df_grid["B2"]) + e))

# 5. Predict with Random Forest for all pixels!
pred_pixels = rf.predict(df_grid[FITUR])
pred_matrix = pred_pixels.reshape(H, W)
print(f"Predicted {len(pred_pixels):,} pixels across East Java!")

# 6. Create RGBA matrix
# Colors matching photo: Red (Bangunan), Gold/Yellow (Sawah), Green (Lahan Hijau), Purple (Mangrove), Dark Blue (Laut), Cyan (Danau)
color_map = {
    1: [255, 200, 30, 215],   # Sawah - Kuning / Gold
    2: [230, 45, 45, 230],    # Bangunan - Merah
    3: [142, 68, 173, 215],   # Mangrove - Ungu
    4: [40, 145, 50, 215],    # Lahan Hijau - Hijau
    5: [25, 75, 160, 220],    # Laut - Biru Tua
    6: [70, 190, 245, 235],   # Danau - Biru Muda
}

rgba_matrix = np.zeros((H, W, 4), dtype=np.uint8)
for k_id, rgba in color_map.items():
    mask = (pred_matrix == k_id)
    rgba_matrix[mask] = rgba

img = Image.fromarray(rgba_matrix, 'RGBA')

# Save PNG
os.makedirs("materi/_static", exist_ok=True)
os.makedirs("materi/output", exist_ok=True)
img.save("materi/_static/peta_lulc_raster.png")
img.save("materi/output/peta_lulc_raster.png")
print("Saved PNG raster overlay successfully.")

# Convert to Base64 data URI for 100% self-contained embedding
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
data_uri = f"data:image/png;base64,{img_b64}"

# 7. Create Folium Map with ImageOverlay
bounds = [[lat_min, lon_min], [lat_max, lon_max]]
pusat = [-7.7, 112.5]

peta = folium.Map(location=pusat, zoom_start=8, control_scale=True)

# Google Satellite
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(peta)

# Esri Satellite
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(peta)

# Add Raster ImageOverlay (LAYER UTAMA SEPERTI DI FOTO!)
ImageOverlay(
    image=data_uri,
    bounds=bounds,
    opacity=0.72,
    name="Klasifikasi Piksel LULC se-Jatim (Raster)",
    interactive=False,
    zindex=4
).add_to(peta)

# Add Bounding Box
grup_aoi = FeatureGroup(name="Batas Area Studi (AOI)", show=True)
folium.Rectangle(bounds, color="yellow", weight=2.5, fill=False, dash_array="6").add_to(grup_aoi)
grup_aoi.add_to(peta)

# Legend HTML
legenda_html = '''
<div style="position:fixed; bottom:30px; left:30px; z-index:9999; background:white;
     padding:12px 14px; border:2px solid #bbb; border-radius:8px; font-size:12px;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
     <b style="font-size:13px;">Klasifikasi LULC Jawa Timur</b><br>
     <div style="margin:3px 0"><span style="display:inline-block;width:13px;height:13px;background:#FFC81E;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>1. Sawah (Pertanian)</div>
     <div style="margin:3px 0"><span style="display:inline-block;width:13px;height:13px;background:#E62D2D;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>2. Bangunan (Permukiman)</div>
     <div style="margin:3px 0"><span style="display:inline-block;width:13px;height:13px;background:#8E44AD;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>3. Mangrove (Pesisir)</div>
     <div style="margin:3px 0"><span style="display:inline-block;width:13px;height:13px;background:#289132;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>4. Lahan Hijau (Hutan)</div>
     <div style="margin:3px 0"><span style="display:inline-block;width:13px;height:13px;background:#194BA0;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>5. Perairan Laut</div>
     <div style="margin:3px 0"><span style="display:inline-block;width:13px;height:13px;background:#46BEF5;border-radius:3px;margin-right:8px;vertical-align:middle;"></span>6. Danau / Waduk</div>
</div>
'''
peta.get_root().html.add_child(folium.Element(legenda_html))
LayerControl(collapsed=False).add_to(peta)
peta.fit_bounds(bounds)

# Save to static files
peta.save("materi/_static/peta_hasil_rf.html")
peta.save("materi/output/peta_hasil_rf.html")
print("Saved HTML map to materi/_static/peta_hasil_rf.html and output.")
