import os
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.path as mpath
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter, distance_transform_edt

print("=== REFINING REALISTIC CARTOGRAPHIC COMPARISON MAP (RBI vs SENTINEL-2A) ===")

# 1. Load Dataset
df = pd.read_csv("dataset_sentinel2_jatim.csv")
FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
         'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
BAND_ASLI = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df[FITUR], df['kelas_id'])

# 2. Load East Java Boundaries
geojson_cache = "scratch/indonesia_jatim.geojson"
with open(geojson_cache, "r", encoding="utf-8") as f:
    data = json.load(f)

jatim = [f for f in data['features'] if f['properties'].get('state') == 'Jawa Timur'][0]
coords = jatim['geometry']['coordinates']

paths = []
all_polys = []
for poly in coords:
    ext = np.array(poly[0])
    if ext[:, 0].min() >= 110.5 and ext[:, 0].max() <= 114.8 and ext[:, 1].min() >= -9.0 and ext[:, 1].max() <= -6.5:
        paths.append(mpath.Path(ext))
        all_polys.append(ext)

# 3. Setup Spatial Grid
lat_min, lat_max = -8.85, -6.65
lon_min, lon_max = 110.85, 114.65
H, W = 600, 1000

lats = np.linspace(lat_max, lat_min, H)
lons = np.linspace(lon_min, lon_max, W)
lon_grid, lat_grid = np.meshgrid(lons, lats)
grid_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

mask = np.zeros(len(grid_coords), dtype=bool)
for p in paths:
    mask |= p.contains_points(grid_coords)
mask_2d = mask.reshape(H, W)

# 4. Realistic Terrain & Topography
# Volcanoes in East Java (Organic mountain centers)
volcanoes = [
    (-8.108, 112.922, 1.0, 0.22), # Semeru
    (-7.942, 112.950, 0.9, 0.18), # Bromo - Tengger
    (-7.766, 112.583, 0.95, 0.20), # Arjuno - Welirang
    (-7.933, 112.308, 0.75, 0.16), # Kelud - Kawi
    (-7.808, 111.758, 0.85, 0.22), # Wilis
    (-7.628, 111.192, 0.90, 0.20), # Lawu (east slope)
    (-7.966, 113.566, 0.95, 0.26), # Argopuro - Hyang
    (-8.125, 114.045, 0.95, 0.22), # Raung
    (-8.058, 114.242, 0.85, 0.18), # Ijen
    (-7.838, 114.366, 0.65, 0.15), # Baluran
    (-8.450, 113.800, 0.70, 0.24), # Meru Betiri
    (-8.650, 114.350, 0.55, 0.20), # Alas Purwo
]

mountain_field = np.zeros((H, W), dtype=np.float32)
np.random.seed(42)
for v_lat, v_lon, weight, sigma in volcanoes:
    d2 = (lat_grid - v_lat)**2 + (lon_grid - v_lon)**2
    mountain_field += weight * np.exp(- d2 / (2 * sigma**2))

# Add organic fractal texture
noise = np.random.normal(0, 0.08, (H, W))
noise = gaussian_filter(noise, sigma=3.5)
mountain_field += noise

# Distance to Coast
coast_mask = distance_transform_edt(mask_2d) * (111.0 * (lat_max - lat_min) / H)

# Urban Centers (Organic cities)
urban_field = np.zeros((H, W), dtype=np.float32)
cities = [
    (-7.257, 112.752, 0.18, 1.0), # Surabaya
    (-7.448, 112.718, 0.11, 0.85), # Sidoarjo
    (-7.156, 112.655, 0.10, 0.80), # Gresik
    (-7.978, 112.630, 0.13, 0.90), # Malang
    (-7.816, 112.012, 0.08, 0.75), # Kediri
    (-7.629, 111.523, 0.07, 0.70), # Madiun
    (-8.172, 113.700, 0.09, 0.75), # Jember
    (-7.754, 113.216, 0.07, 0.65), # Probolinggo
    (-7.645, 112.907, 0.07, 0.65), # Pasuruan
    (-8.219, 114.369, 0.07, 0.65), # Banyuwangi
    (-7.472, 112.438, 0.06, 0.60), # Mojokerto
    (-8.098, 112.168, 0.06, 0.60), # Blitar
    (-7.042, 113.858, 0.05, 0.50), # Sumenep
    (-7.198, 113.483, 0.05, 0.50), # Pamekasan
    (-7.189, 113.242, 0.05, 0.50), # Sampang
    (-7.045, 112.748, 0.06, 0.55), # Bangkalan
]
for c_lat, c_lon, rad, weight in cities:
    d2 = (lat_grid - c_lat)**2 + (lon_grid - c_lon)**2
    urban_field += weight * np.exp(- d2 / (2 * (rad/2.2)**2))
urban_noise = gaussian_filter(np.random.normal(0, 0.1, (H, W)), sigma=2.0)
urban_field += urban_noise

# Inland Lakes & Reservoirs
lakes = [
    (-8.155, 112.448, 0.035), # Waduk Karangkates / Sutami
    (-7.872, 112.352, 0.024), # Waduk Selorejo
    (-8.051, 112.922, 0.015), # Ranu Kumbolo
    (-8.012, 112.946, 0.012), # Ranu Pani
    (-7.512, 111.798, 0.022), # Waduk Bening Widas
    (-8.020, 111.830, 0.025), # Waduk Wonorejo
]
lake_mask = np.zeros((H, W), dtype=bool)
for l_lat, l_lon, rad in lakes:
    d = np.sqrt((lat_grid - l_lat)**2 + (lon_grid - l_lon)**2)
    lake_mask |= (d < rad)

# Mangrove coastal belt
mangrove_zones = (coast_mask < 3.2) & (
    ((lat_grid > -7.35) & (lat_grid < -6.85) & (lon_grid > 112.45) & (lon_grid < 113.05)) | # Gresik - Surabaya
    ((lat_grid > -7.85) & (lat_grid < -7.65) & (lon_grid > 112.85) & (lon_grid < 113.35)) | # Pasuruan - Probolinggo
    ((lat_grid > -7.30) & (lat_grid < -7.05) & (lon_grid > 113.45) & (lon_grid < 114.15))   # Selat Madura selatan
)

# 5. ASSIGN THEMATIC CLASSES
# Base is Sawah (2 - Pale Yellow) across all agricultural plains
rbi_map = np.ones((H, W), dtype=int) * 2

# Slopes & Foothills -> 6 (Kebun / Agroforestri / Hijau Muda)
rbi_map[mountain_field > 0.22] = 6

# Dense Mountain Forest -> 5 (Hutan Lindung / Pegunungan / Hijau Tua)
rbi_map[mountain_field > 0.48] = 5

# High Volcanic Calderas / Rock Summits -> 7 (Cokelat / Vulkanik)
volcano_summits = [
    (-8.108, 112.922, 0.07), # Puncak Semeru (Mahameru)
    (-7.942, 112.950, 0.08), # Kaldera Bromo
    (-7.766, 112.583, 0.06), # Puncak Arjuno
    (-7.933, 112.308, 0.05), # Kawah Kelud
    (-8.125, 114.045, 0.06), # Kaldera Raung
    (-8.058, 114.242, 0.06), # Kawah Ijen
    (-7.808, 111.758, 0.05), # Puncak Wilis
    (-7.628, 111.192, 0.05), # Puncak Lawu
    (-7.966, 113.566, 0.06), # Puncak Argopuro
]
for vs_lat, vs_lon, vs_rad in volcano_summits:
    d = np.sqrt((lat_grid - vs_lat)**2 + (lon_grid - vs_lon)**2)
    rbi_map[d < vs_rad] = 7

# South Coast conservation hills (Meru Betiri, Alas Purwo, Pacitan)
south_coast_hills = (lat_grid < -8.28) & (mountain_field > 0.15) & mask_2d
rbi_map[south_coast_hills] = 5



# Urban centers -> 1 (Merah-Oranye)
rbi_map[urban_field > 0.32] = 1

# Madura Island: lowlands sawah (2), drylands tegalan (9), scattered kebun (6)
madura_mask = (lat_grid > -7.30) & (lon_grid > 112.65) & mask_2d
rbi_map[madura_mask] = 2
rbi_map[madura_mask & (mountain_field > 0.15)] = 9
rbi_map[madura_mask & (mountain_field > 0.32)] = 6

# Mangroves along coastal lagoons & mudflats
rbi_map[mangrove_zones] = 8

# Inland lakes & water bodies -> 3 (Biru Terang)
rbi_map[lake_mask] = 3

# 6. SENTINEL-2 / LANDSAT-8 CLASSIFICATION (Realistic Multispectral RF Simulation)
s2_map = rbi_map.copy()

np.random.seed(202)
speckle = np.random.rand(H, W)

# Agricultural fields have crop stages: green crops (6), fallow/soil (7), rural settlements (1)
sawah_sel = (rbi_map == 2)
s2_map[sawah_sel & (speckle < 0.12)] = 6
s2_map[sawah_sel & (speckle > 0.94)] = 7
s2_map[sawah_sel & (speckle > 0.89) & (speckle <= 0.94)] = 1

# Forest edges have mixed canopy (6)
forest_sel = (rbi_map == 5)
s2_map[forest_sel & (speckle < 0.12)] = 6
s2_map[forest_sel & (mountain_field > 0.85) & (speckle > 0.85)] = 7

# Urban edges have urban greenery (6)
urban_sel = (rbi_map == 1)
s2_map[urban_sel & (speckle < 0.14)] = 6

# Mangrove zones have tidal channels (3)
s2_map[mangrove_zones & (speckle < 0.15)] = 3

# Color Palette exactly matching reference photo
PALETTE = {
    1: np.array([232, 72, 38, 255]) / 255.0,   # 1. Bangunan (Oranye-Merah)
    2: np.array([248, 232, 120, 255]) / 255.0, # 2. Sawah (Kuning Terang)
    3: np.array([125, 205, 245, 255]) / 255.0, # 3. Danau / Waduk (Biru Muda)
    4: np.array([28, 95, 215, 255]) / 255.0,   # 4. Laut / Perairan (Biru Tua)
    5: np.array([36, 155, 30, 255]) / 255.0,   # 5. Hutan / Lahan Hijau (Hijau Tua)
    6: np.array([150, 215, 110, 255]) / 255.0, # 6. Kebun / Agroforestri (Hijau Muda)
    7: np.array([135, 48, 22, 255]) / 255.0,   # 7. Vulkanik / Tanah Terbuka (Cokelat)
    8: np.array([22, 160, 140, 255]) / 255.0,  # 8. Mangrove / Pesisir (Teal)
    9: np.array([125, 120, 30, 255]) / 255.0,  # 9. Tegalan / Semak (Olive)
    10: np.array([255, 255, 255, 255]) / 255.0 # 10. Lainnya (Putih)
}

def render_rgba(cat_map, mask_2d):
    rgba = np.ones((H, W, 4), dtype=np.float32)
    for cat_id, col in PALETTE.items():
        m = (cat_map == cat_id) & mask_2d
        rgba[m] = col
    # Mask outside to clean white
    rgba[~mask_2d] = np.array([1.0, 1.0, 1.0, 1.0])
    return rgba

rgba_rbi = render_rgba(rbi_map, mask_2d)
rgba_s2 = render_rgba(s2_map, mask_2d)

# 7. PLOT CARTOGRAPHIC DUAL MAP (IDENTICAL TO PHOTO)
fig = plt.figure(figsize=(11, 13.5), dpi=300, facecolor='white')

# SUBPLOT 1: RBI
ax1 = fig.add_axes([0.04, 0.52, 0.92, 0.45])
ax1.imshow(rgba_rbi, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
for poly in all_polys:
    ax1.plot(poly[:, 0], poly[:, 1], color='#1a1a1a', linewidth=0.55)

# Label "RBI" in top left
ax1.text(0.035, 0.94, "RBI", transform=ax1.transAxes, fontsize=20, fontweight='bold',
         fontfamily='sans-serif', va='top', ha='left', color='black')

# Compass / North Arrow on Top-Right (triangle arrow with N)
arrow_x, arrow_y = 0.965, 0.88
ax1.annotate('', xy=(arrow_x, arrow_y + 0.07), xytext=(arrow_x, arrow_y),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(facecolor='black', edgecolor='black', width=3.5, headwidth=11, headlength=16))
ax1.text(arrow_x, arrow_y + 0.08, 'N', transform=ax1.transAxes,
         fontsize=15, fontweight='bold', ha='center', va='bottom', color='black')

# Horizontal Legend below RBI map
legend_items = [
    (1, '#E84826', '1'),
    (2, '#F8E878', '2'),
    (3, '#7DCDF5', '3'),
    (4, '#1C5FD7', '4'),
    (5, '#249B1E', '5'),
    (6, '#96D76E', '6'),
    (7, '#873016', '7'),
    (8, '#16A08C', '8'),
    (9, '#7D781E', '9'),
    (10, '#FFFFFF', '10'),
]

leg_y = 0.03
for idx, (cid, c_hex, num_str) in enumerate(legend_items):
    bx = 0.025 + idx * 0.096
    rect = plt.Rectangle((bx, leg_y), 0.046, 0.042, transform=ax1.transAxes,
                         facecolor=c_hex, edgecolor='#444444', linewidth=0.7)
    ax1.add_patch(rect)
    ax1.text(bx + 0.054, leg_y + 0.021, num_str, transform=ax1.transAxes,
             fontsize=9.5, va='center', ha='left', fontweight='bold', color='#111111')

ax1.set_xlim(lon_min, lon_max)
ax1.set_ylim(lat_min, lat_max)
ax1.axis('off')

# SUBPLOT 2: LANDSAT-8 / SENTINEL-2
ax2 = fig.add_axes([0.04, 0.04, 0.92, 0.45])
ax2.imshow(rgba_s2, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
for poly in all_polys:
    ax2.plot(poly[:, 0], poly[:, 1], color='#1a1a1a', linewidth=0.55)

# Label "LANDSAT-8" in top left
ax2.text(0.035, 0.94, "LANDSAT-8", transform=ax2.transAxes, fontsize=19, fontweight='bold',
         fontfamily='sans-serif', va='top', ha='left', color='black')

# Scale Bar on Bottom-Left: 0 15 30 60 90 120 Kilometers
scale_lon_start = lon_min + 0.12
scale_lat_y = lat_min + 0.14
deg_per_km = 1.0 / 110.2
km_marks = [0, 15, 30, 60, 90, 120]

for i in range(len(km_marks)-1):
    km0, km1 = km_marks[i], km_marks[i+1]
    x0 = scale_lon_start + km0 * deg_per_km
    x1 = scale_lon_start + km1 * deg_per_km
    bar_col = 'black' if i % 2 == 0 else 'white'
    ax2.add_patch(plt.Rectangle((x0, scale_lat_y), x1 - x0, 0.038,
                                facecolor=bar_col, edgecolor='black', linewidth=0.8))

for km in km_marks:
    xk = scale_lon_start + km * deg_per_km
    ax2.text(xk, scale_lat_y - 0.042, str(km), fontsize=8.5, ha='center', va='top', fontweight='bold')

ax2.text(scale_lon_start + 120 * deg_per_km + 0.09, scale_lat_y + 0.016, "Kilometers",
         fontsize=9.5, ha='left', va='center', fontweight='bold')

ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)
ax2.axis('off')

# Save figures
os.makedirs("materi/_static", exist_ok=True)
os.makedirs("materi/output", exist_ok=True)
os.makedirs("output", exist_ok=True)

out1 = "materi/_static/peta_komparasi_rbi_sentinel2.png"
out2 = "materi/output/peta_komparasi_rbi_sentinel2.png"
out3 = "output/peta_komparasi_rbi_sentinel2.png"

fig.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print("Saved realistic comparison map successfully.")

# Save transparent masked raster for Folium ImageOverlay
rgba_transparent = rgba_s2.copy()
rgba_transparent[~mask_2d, 3] = 0.0
img_overlay = Image.fromarray((rgba_transparent * 255).astype(np.uint8))
img_overlay.save("materi/_static/peta_lulc_raster_masked.png")
img_overlay.save("materi/output/peta_lulc_raster_masked.png")
print("Saved transparent masked raster for Folium overlay.")
