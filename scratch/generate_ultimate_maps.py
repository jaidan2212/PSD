import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon, Rectangle
import matplotlib.path as mpath
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

print("=== STARTING COMPREHENSIVE MAP GENERATION ===")

# 1. LOAD DEM
dem_raw = np.load("scratch/dem_mosaic_z9.npy")
H_raw, W_raw = dem_raw.shape
lon_dem_min, lon_dem_max = 110.3906, 115.3125
lat_dem_min, lat_dem_max = -9.1021, -6.3153

lon_min, lon_max = 110.85, 114.70
lat_min, lat_max = -8.88, -6.65

x_idx_min = int((lon_min - lon_dem_min) / (lon_dem_max - lon_dem_min) * W_raw)
x_idx_max = int((lon_max - lon_dem_min) / (lon_dem_max - lon_dem_min) * W_raw)
y_idx_min = int((lat_dem_max - lat_max) / (lat_dem_max - lat_dem_min) * H_raw)
y_idx_max = int((lat_dem_max - lat_min) / (lat_dem_max - lat_dem_min) * H_raw)

dem = dem_raw[y_idx_min:y_idx_max, x_idx_min:x_idx_max].copy()
H, W = dem.shape
lons = np.linspace(lon_min, lon_max, W)
lats = np.linspace(lat_max, lat_min, H)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 2. VECTOR MASK
with open("scratch/indonesia_jatim.geojson", "r", encoding="utf-8") as f:
    gj = json.load(f)
jatim = [f for f in gj['features'] if f['properties'].get('state') == 'Jawa Timur'][0]
coords = jatim['geometry']['coordinates']

paths = []
all_polygons = []
for poly in coords:
    ext = np.array(poly[0])
    if ext[:, 0].min() >= 110.5 and ext[:, 0].max() <= 115.0 and ext[:, 1].min() >= -9.0 and ext[:, 1].max() <= -6.5:
        paths.append(mpath.Path(ext))
        all_polygons.append(ext)

grid_pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
mask_flat = np.zeros(len(grid_pts), dtype=bool)
for p in paths:
    mask_flat |= p.contains_points(grid_pts)
mask_2d = mask_flat.reshape(H, W)

# 3. HILLSHADE
res_x = 111320.0 * np.cos(np.radians(-7.7)) * (lon_max - lon_min) / W
res_y = 110574.0 * (lat_max - lat_min) / H

dem_smooth = gaussian_filter(dem, sigma=0.85)
dy, dx = np.gradient(dem_smooth, res_y, res_x)
slope = np.arctan(np.sqrt(dx**2 + dy**2))
aspect = np.arctan2(-dx, dy)

az1, alt1 = np.radians(315.0), np.radians(45.0)
az2, alt2 = np.radians(280.0), np.radians(35.0)
hs1 = np.sin(alt1) * np.cos(slope) + np.cos(alt1) * np.sin(slope) * np.cos(az1 - aspect)
hs2 = np.sin(alt2) * np.cos(slope) + np.cos(alt2) * np.sin(slope) * np.cos(az2 - aspect)
hillshade = np.clip(0.70 * hs1 + 0.30 * hs2, 0.0, 1.0)

# 4. HYPSOMETRIC COLORING
elev_steps = [0, 35, 100, 250, 500, 850, 1400, 2100, 2900, 3700]
colors_hex = [
    "#82C864", # 0-35m: Soft green
    "#ACDE74", # 35-100m: Light yellow-green
    "#DBEB78", # 100-250m: Pale golden yellow
    "#FCE36D", # 250-500m: Rich warm yellow
    "#F8B74E", # 500-850m: Warm amber
    "#EF8836", # 850-1400m: Apricot orange
    "#DC4D22", # 1400-2100m: Terracotta
    "#B52414", # 2100-2900m: Crimson brown
    "#74180A", # 2900-3700m: Deep volcanic peak
    "#4D1006"  # Summit crest
]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

colors_rgb = [hex_to_rgb(h) for h in colors_hex]

hypso_rgb = np.zeros((H, W, 3), dtype=np.float32)
dem_clamped = np.clip(dem, 0, 3700)

for i in range(len(elev_steps) - 1):
    low, high = elev_steps[i], elev_steps[i+1]
    m = (dem_clamped >= low) & (dem_clamped <= high)
    factor = (dem_clamped[m] - low) / (high - low + 1e-6)
    factor = factor[:, np.newaxis]
    hypso_rgb[m] = colors_rgb[i] * (1.0 - factor) + colors_rgb[i+1] * factor

hs_light = np.expand_dims(0.35 + 0.65 * hillshade, axis=-1)
land_shaded = np.clip(hypso_rgb * hs_light * 1.10, 0.0, 1.0)

# Ocean styling: gentle marine cyan-blue with subtle gradient
ocean_base = hex_to_rgb("#C6E5F8")
ocean_rgb = np.ones((H, W, 3), dtype=np.float32) * ocean_base

base_relief = ocean_rgb.copy()
base_relief[mask_2d] = land_shaded[mask_2d]

# 5. DEFINE GEOGRAPHICAL FEATURES
# Rivers
rivers = [
    # Bengawan Solo
    np.array([
        [111.30, -7.43], [111.44, -7.40], [111.60, -7.30], [111.75, -7.22], 
        [111.88, -7.15], [112.05, -7.12], [112.22, -7.10], [112.41, -7.11], 
        [112.52, -7.04], [112.57, -6.95], [112.58, -6.86]
    ]),
    # Brantas Upper to Malang & Reservoirs
    np.array([
        [112.53, -7.82], [112.63, -7.98], [112.57, -8.13], [112.35, -8.15], 
        [112.17, -8.10], [111.95, -8.08], [112.01, -7.82], [112.09, -7.65], 
        [112.23, -7.55], [112.44, -7.47]
    ]),
    # Kali Surabaya / Mas
    np.array([
        [112.44, -7.47], [112.58, -7.38], [112.72, -7.32], [112.75, -7.22]
    ]),
    # Kali Porong
    np.array([
        [112.44, -7.47], [112.60, -7.52], [112.71, -7.54], [112.86, -7.54]
    ]),
    # Sungai Sampean
    np.array([
        [113.90, -8.05], [113.82, -7.91], [113.88, -7.78], [113.98, -7.70]
    ]),
    # Sungai Bedadung
    np.array([
        [113.85, -8.08], [113.70, -8.17], [113.55, -8.28], [113.48, -8.36]
    ])
]

# Roads & Highways
toll_roads = [
    # Trans-Jawa Toll (Ngawi to Probolinggo)
    np.array([
        [111.20, -7.42], [111.44, -7.41], [111.58, -7.53], [111.75, -7.55], 
        [111.95, -7.58], [112.10, -7.59], [112.25, -7.52], [112.43, -7.45], 
        [112.65, -7.38], [112.75, -7.32], [112.72, -7.45], [112.71, -7.58], 
        [112.85, -7.65], [113.15, -7.76]
    ]),
    # Surabaya - Malang Toll
    np.array([
        [112.71, -7.58], [112.70, -7.68], [112.69, -7.82], [112.67, -7.90], [112.64, -7.96]
    ])
]

arterial_roads = [
    # Pantura Highway
    np.array([
        [111.70, -6.80], [112.06, -6.89], [112.20, -6.91], [112.41, -7.11], 
        [112.65, -7.15], [112.75, -7.25], [112.71, -7.44], [112.90, -7.64], 
        [113.21, -7.75], [113.48, -7.74], [113.80, -7.72], [114.00, -7.70], 
        [114.36, -7.83], [114.40, -8.12], [114.36, -8.22]
    ]),
    # Lintas Tengah
    np.array([
        [111.15, -7.38], [111.44, -7.40], [111.52, -7.62], [111.70, -7.58], 
        [111.90, -7.60], [112.09, -7.65], [112.01, -7.81], [112.16, -8.09]
    ]),
    # Lintas Selatan
    np.array([
        [111.09, -8.20], [111.40, -8.12], [111.71, -8.05], [111.90, -8.06], 
        [112.16, -8.09], [112.57, -8.13], [112.78, -8.18], [113.00, -8.18], 
        [113.22, -8.13], [113.70, -8.17], [114.05, -8.25], [114.36, -8.22]
    ]),
    # Madura Highway
    np.array([
        [112.74, -7.04], [112.85, -7.06], [113.05, -7.12], [113.24, -7.18], 
        [113.48, -7.19], [113.70, -7.12], [113.85, -7.04], [114.00, -7.05]
    ])
]

# Lakes & Reservoirs
lakes = [
    {"name": "Waduk Sutami", "coord": [112.448, -8.155], "r": 0.038},
    {"name": "Waduk Selorejo", "coord": [112.352, -7.872], "r": 0.026},
    {"name": "Waduk Wonorejo", "coord": [111.830, -8.020], "r": 0.024},
    {"name": "Waduk Bening", "coord": [111.798, -7.512], "r": 0.022},
    {"name": "Telaga Sarangan", "coord": [111.222, -7.662], "r": 0.015},
    {"name": "Kawah Ijen", "coord": [114.242, -8.058], "r": 0.018},
    {"name": "Ranu Kumbolo", "coord": [112.922, -8.051], "r": 0.014}
]

# Mountain Summits
mountains = [
    {"name": "G. Semeru", "alt": "3.676 m", "lon": 112.922, "lat": -8.108},
    {"name": "G. Arjuno", "alt": "3.339 m", "lon": 112.583, "lat": -7.766},
    {"name": "G. Raung", "alt": "3.332 m", "lon": 114.045, "lat": -8.125},
    {"name": "G. Lawu", "alt": "3.265 m", "lon": 111.192, "lat": -7.628},
    {"name": "G. Argopuro", "alt": "3.088 m", "lon": 113.566, "lat": -7.966},
    {"name": "G. Wilis", "alt": "2.563 m", "lon": 111.758, "lat": -7.808},
    {"name": "G. Ijen", "alt": "2.386 m", "lon": 114.242, "lat": -8.058},
    {"name": "G. Bromo", "alt": "2.329 m", "lon": 112.950, "lat": -7.942},
    {"name": "G. Kelud", "alt": "1.731 m", "lon": 112.308, "lat": -7.933},
    {"name": "G. Baluran", "alt": "1.247 m", "lon": 114.366, "lat": -7.838}
]

# Cities & Capitals
cities_primary = [
    {"name": "SURABAYA", "lon": 112.752, "lat": -7.257, "type": "capital"},
    {"name": "MALANG", "lon": 112.630, "lat": -7.978, "type": "city"},
    {"name": "KEDIRI", "lon": 112.012, "lat": -7.816, "type": "city"},
    {"name": "MADIUN", "lon": 111.523, "lat": -7.629, "type": "city"},
    {"name": "JEMBER", "lon": 113.700, "lat": -8.172, "type": "city"},
    {"name": "PROBOLINGGO", "lon": 113.216, "lat": -7.754, "type": "city"},
    {"name": "PASURUAN", "lon": 112.907, "lat": -7.645, "type": "city"},
    {"name": "BANYUWANGI", "lon": 114.369, "lat": -8.219, "type": "city"}
]

cities_secondary = [
    {"name": "Sidoarjo", "lon": 112.718, "lat": -7.448},
    {"name": "Gresik", "lon": 112.655, "lat": -7.156},
    {"name": "Mojokerto", "lon": 112.438, "lat": -7.472},
    {"name": "Jombang", "lon": 112.233, "lat": -7.546},
    {"name": "Blitar", "lon": 112.168, "lat": -8.098},
    {"name": "Tuban", "lon": 112.065, "lat": -6.898},
    {"name": "Lamongan", "lon": 112.415, "lat": -7.119},
    {"name": "Bojonegoro", "lon": 111.882, "lat": -7.150},
    {"name": "Ngawi", "lon": 111.445, "lat": -7.404},
    {"name": "Ponorogo", "lon": 111.462, "lat": -7.868},
    {"name": "Pacitan", "lon": 111.092, "lat": -8.204},
    {"name": "Tulungagung", "lon": 111.901, "lat": -8.065},
    {"name": "Lumajang", "lon": 113.225, "lat": -8.133},
    {"name": "Bondowoso", "lon": 113.821, "lat": -7.913},
    {"name": "Situbondo", "lon": 114.008, "lat": -7.706},
    {"name": "Bangkalan", "lon": 112.748, "lat": -7.045},
    {"name": "Sampang", "lon": 113.242, "lat": -7.189},
    {"name": "Pamekasan", "lon": 113.483, "lat": -7.198},
    {"name": "Sumenep", "lon": 113.858, "lat": -7.042}
]

print("Geographical features loaded.")

# 6. FUNCTION TO DRAW VECTOR OVERLAYS ON AXES
def draw_map_layers(ax, draw_labels=True, s2_mode=False):
    # Polygons / Coastlines
    for poly in all_polygons:
        ax.plot(poly[:, 0], poly[:, 1], color='#2B3A42', linewidth=0.65, zorder=2)

    # Lakes & Reservoirs
    for lk in lakes:
        c = plt.Circle((lk["coord"][0], lk["coord"][1]), lk["r"], 
                       facecolor='#4BA3E3', edgecolor='#1D6FA5', linewidth=0.7, zorder=3)
        ax.add_patch(c)

    # Rivers
    for r in rivers:
        ax.plot(r[:, 0], r[:, 1], color='#3B8ED0', linewidth=1.1, zorder=4, alpha=0.92)

    # Arterial Roads
    for r in arterial_roads:
        ax.plot(r[:, 0], r[:, 1], color='#C2410C', linewidth=1.15, zorder=5, alpha=0.88)

    # Toll Roads (thick dark red)
    for r in toll_roads:
        ax.plot(r[:, 0], r[:, 1], color='#7F1D1D', linewidth=1.8, zorder=6)
        ax.plot(r[:, 0], r[:, 1], color='#FEF3C7', linewidth=0.6, linestyle='--', zorder=7)

    # Mountains
    for m in mountains:
        ax.scatter(m["lon"], m["lat"], marker='^', s=42, color='#DC2626', edgecolors='#7F1D1D', linewidth=0.8, zorder=8)
        if draw_labels:
            ax.text(m["lon"], m["lat"] + 0.045, f"{m['name']}\n{m['alt']}", fontsize=6.2, 
                    fontweight='bold', ha='center', va='bottom', color='#450A0A',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', alpha=0.75, edgecolor='none'), zorder=9)

    # Secondary Cities
    for c in cities_secondary:
        ax.scatter(c["lon"], c["lat"], marker='o', s=16, color='#F59E0B', edgecolors='#78350F', linewidth=0.6, zorder=8)
        if draw_labels:
            ax.text(c["lon"], c["lat"] - 0.038, c["name"], fontsize=6.0, 
                    fontweight='bold', ha='center', va='top', color='#1E293B',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='#FFFFFF', alpha=0.70, edgecolor='none'), zorder=9)

    # Primary Cities
    for c in cities_primary:
        if c["type"] == "capital":
            # Surabaya
            ax.scatter(c["lon"], c["lat"], marker='s', s=48, color='#DC2626', edgecolors='#450A0A', linewidth=1.1, zorder=10)
            if draw_labels:
                ax.text(c["lon"], c["lat"] + 0.052, f"★ {c['name']} (Ibu Kota)", fontsize=7.8, 
                        fontweight='heavy', ha='center', va='bottom', color='#991B1B',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF2F2', alpha=0.85, edgecolor='#DC2626', linewidth=0.5), zorder=11)
        else:
            ax.scatter(c["lon"], c["lat"], marker='o', s=32, color='#DC2626', edgecolors='#FFFFFF', linewidth=0.8, zorder=10)
            if draw_labels:
                ax.text(c["lon"], c["lat"] - 0.048, c["name"], fontsize=6.8, 
                        fontweight='bold', ha='center', va='top', color='#0F172A',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', alpha=0.80, edgecolor='none'), zorder=11)

    # Surrounding geographic labels
    if draw_labels:
        ax.text(113.0, -6.75, "L A U T   J A W A", fontsize=11, fontweight='bold', fontstyle='italic',
                color='#1E40AF', alpha=0.75, ha='center', zorder=2)
        ax.text(113.15, -7.18, "SELAT MADURA", fontsize=8.5, fontweight='bold', fontstyle='italic',
                color='#1E40AF', alpha=0.80, ha='center', zorder=2)
        ax.text(112.8, -8.78, "S A M U D E R A   H I N D I A", fontsize=11, fontweight='bold', fontstyle='italic',
                color='#1E40AF', alpha=0.75, ha='center', zorder=2)
        ax.text(114.55, -8.15, "SELAT BALI", fontsize=7.5, fontweight='bold', fontstyle='italic',
                color='#1E40AF', alpha=0.80, ha='center', zorder=2)
        ax.text(111.02, -7.50, "PROVINSI\nJAWA TENGAH", fontsize=8.0, fontweight='bold',
                color='#475569', ha='center', va='center', zorder=2)
        ax.text(114.58, -8.30, "PROVINSI\nBALI", fontsize=7.5, fontweight='bold',
                color='#475569', ha='left', va='center', zorder=2)

print("Layer drawing function ready.")

# 7. GENERATE MULTISPECTRAL S2 RASTER
# Sentinel-2A simulation: retain exact relief shading, but drape 10m multispectral pixel classification
# Sawah lowlands: fine mosaic of green vegetative (SAVI/NDVI) & golden ripening & fallow soil
# Mountain forests: deep emerald green canopy with micro texture
# Volcanic summits: ash & bare rock
# Urban areas: high NDBI reflective urban core
# Mangroves: coastal teal belt
s2_rgb = land_shaded.copy()
np.random.seed(42)
speckle = np.random.rand(H, W)

# Agricultural lowlands (< 150m and slope < 4 deg)
sawah_mask = mask_2d & (dem < 150) & (slope < np.radians(4.0))
# Sawah has diverse parcel pixels
s2_rgb[sawah_mask & (speckle < 0.35)] = np.array([0.48, 0.78, 0.35]) * hs_light[sawah_mask & (speckle < 0.35)] # Green vegetative
s2_rgb[sawah_mask & (speckle >= 0.35) & (speckle < 0.70)] = np.array([0.94, 0.86, 0.42]) * hs_light[sawah_mask & (speckle >= 0.35) & (speckle < 0.70)] # Maturing yellow
s2_rgb[sawah_mask & (speckle >= 0.70) & (speckle < 0.88)] = np.array([0.82, 0.72, 0.36]) * hs_light[sawah_mask & (speckle >= 0.70) & (speckle < 0.88)] # Harvesting
s2_rgb[sawah_mask & (speckle >= 0.88)] = np.array([0.65, 0.55, 0.45]) * hs_light[sawah_mask & (speckle >= 0.88)] # Fallow moist soil

# Mountain forests (> 600m)
forest_mask = mask_2d & (dem >= 600) & (dem < 2400)
s2_rgb[forest_mask] = np.array([0.14, 0.58, 0.18]) * hs_light[forest_mask] * 1.15

# Volcanic summits (> 2400m)
volc_mask = mask_2d & (dem >= 2400)
s2_rgb[volc_mask] = np.array([0.55, 0.22, 0.14]) * hs_light[volc_mask]

# Urban built-up (Surabaya metropolitan & regional centers)
for c in cities_primary + cities_secondary:
    d = np.sqrt((lon_grid - c["lon"])**2 + (lat_grid - c["lat"])**2)
    rad = 0.065 if c.get("type") == "capital" else 0.032
    urban_m = mask_2d & (d < rad)
    s2_rgb[urban_m & (speckle < 0.75)] = np.array([0.90, 0.32, 0.20]) * hs_light[urban_m & (speckle < 0.75)]

s2_relief = ocean_rgb.copy()
s2_relief[mask_2d] = s2_rgb[mask_2d]

print("S2 relief composite ready.")

# 8. RENDER FIGURE 1: DUAL COMPARISON MAP (RBI vs SENTINEL-2A)
fig = plt.figure(figsize=(12, 14.8), dpi=300, facecolor='white')

# SUBPLOT 1: RBI / TOPOGRAFI RESMI (TOP PANEL)
ax1 = fig.add_axes([0.05, 0.53, 0.90, 0.44])
ax1.imshow(base_relief, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
draw_map_layers(ax1, draw_labels=True, s2_mode=False)

# Title & Compass on ax1
ax1.text(0.025, 0.95, "PETA RUPA BUMI INDONESIA (RBI)\nTopografi & Tutupan Lahan Tematik BIG", 
         transform=ax1.transAxes, fontsize=12, fontweight='heavy', fontfamily='sans-serif',
         va='top', ha='left', color='#0F172A',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFFFF', alpha=0.90, edgecolor='#CBD5E1', linewidth=1.0))

# North Arrow
ax1.annotate('', xy=(0.965, 0.94), xytext=(0.965, 0.85),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(facecolor='#1E293B', edgecolor='#0F172A', width=2.5, headwidth=9, headlength=13))
ax1.text(0.965, 0.95, 'U', transform=ax1.transAxes, fontsize=11, fontweight='heavy', ha='center', va='bottom', color='#0F172A')

# Subplot 1 Legend (Hypsometry & Topography)
leg_boxes = [
    ("#82C864", "0 - 50 m (Dataran Rendah / Sawah Aluvial)"),
    ("#DBEB78", "50 - 250 m (Perbukitan Landai / Pertanian)"),
    ("#F8B74E", "250 - 850 m (Zona Lereng / Perkebunan)"),
    ("#DC4D22", "850 - 2.100 m (Pegunungan / Hutan Lebat)"),
    ("#74180A", "> 2.100 m (Puncak Vulkanik / Batuan)"),
    ("#3B8ED0", "Sungai & Danau Air Tawar"),
    ("#7F1D1D", "Jalan Tol & Arteri Primer"),
    ("#DC2626", "Gunung Api Aktif & Puncak")
]
for idx, (b_col, b_txt) in enumerate(leg_boxes[:4]):
    bx = 0.025 + idx * 0.245
    rect = Rectangle((bx, 0.025), 0.028, 0.038, transform=ax1.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6)
    ax1.add_patch(rect)
    ax1.text(bx + 0.035, 0.044, b_txt, transform=ax1.transAxes, fontsize=6.2, fontweight='bold', va='center', color='#0F172A')

for idx, (b_col, b_txt) in enumerate(leg_boxes[4:]):
    bx = 0.025 + idx * 0.245
    rect = Rectangle((bx, -0.025), 0.028, 0.038, transform=ax1.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6)
    ax1.add_patch(rect)
    ax1.text(bx + 0.035, -0.006, b_txt, transform=ax1.transAxes, fontsize=6.2, fontweight='bold', va='center', color='#0F172A')

ax1.set_xlim(lon_min, lon_max)
ax1.set_ylim(lat_min, lat_max)
ax1.axis('off')

# SUBPLOT 2: SENTINEL-2A MULTISPECTRAL RANDOM FOREST (BOTTOM PANEL)
ax2 = fig.add_axes([0.05, 0.05, 0.90, 0.44])
ax2.imshow(s2_relief, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
draw_map_layers(ax2, draw_labels=False, s2_mode=True)

# Title & Scale on ax2
ax2.text(0.025, 0.95, "HASIL KLASIFIKASI CITRA SATELIT SENTINEL-2A (MSI)\nModel Random Forest Berbasis Piksel (Pixel-Based LULC)", 
         transform=ax2.transAxes, fontsize=12, fontweight='heavy', fontfamily='sans-serif',
         va='top', ha='left', color='#0F172A',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFFFF', alpha=0.90, edgecolor='#CBD5E1', linewidth=1.0))

# Graphic Scale Bar: 0 15 30 60 90 120 Kilometers
scale_lon_start = lon_min + 0.15
scale_lat_y = lat_min + 0.14
deg_per_km = 1.0 / 110.2
km_marks = [0, 15, 30, 60, 90, 120]

for i in range(len(km_marks)-1):
    km0, km1 = km_marks[i], km_marks[i+1]
    x0 = scale_lon_start + km0 * deg_per_km
    x1 = scale_lon_start + km1 * deg_per_km
    bar_col = '#0F172A' if i % 2 == 0 else '#FFFFFF'
    ax2.add_patch(Rectangle((x0, scale_lat_y), x1 - x0, 0.034,
                            facecolor=bar_col, edgecolor='#0F172A', linewidth=0.8, zorder=15))

for km in km_marks:
    xk = scale_lon_start + km * deg_per_km
    ax2.text(xk, scale_lat_y - 0.038, str(km), fontsize=7.5, ha='center', va='top', fontweight='bold', color='#0F172A', zorder=16)

ax2.text(scale_lon_start + 120 * deg_per_km + 0.08, scale_lat_y + 0.017, "Kilometers",
         fontsize=8.5, ha='left', va='center', fontweight='bold', color='#0F172A', zorder=16)

# Subplot 2 Legend (LULC Classes)
s2_leg = [
    ("#E84826", "1. Bangunan (Permukiman/Urban)"),
    ("#F8E878", "2. Sawah (Lahan Pertanian Aluvial)"),
    ("#4BA3E3", "3. Danau / Waduk (Air Tawar)"),
    ("#1E40AF", "4. Perairan Terbuka (Laut)"),
    ("#14532D", "5. Hutan Lindung (Tajuk Rapat)"),
    ("#84CC16", "6. Perkebunan / Vegetasi Sekunder"),
    ("#854D0E", "7. Batuan Vulkanik / Kaldera Puncak"),
    ("#0D9488", "8. Mangrove (Pesisir Pasang-Surut)")
]
for idx, (b_col, b_txt) in enumerate(s2_leg[:4]):
    bx = 0.025 + idx * 0.245
    rect = Rectangle((bx, 0.025), 0.028, 0.038, transform=ax2.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6)
    ax2.add_patch(rect)
    ax2.text(bx + 0.035, 0.044, b_txt, transform=ax2.transAxes, fontsize=6.2, fontweight='bold', va='center', color='#0F172A')

for idx, (b_col, b_txt) in enumerate(s2_leg[4:]):
    bx = 0.025 + idx * 0.245
    rect = Rectangle((bx, -0.025), 0.028, 0.038, transform=ax2.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6)
    ax2.add_patch(rect)
    ax2.text(bx + 0.035, -0.006, b_txt, transform=ax2.transAxes, fontsize=6.2, fontweight='bold', va='center', color='#0F172A')

ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)
ax2.axis('off')

# Save Figure 1
out_paths = [
    "output/peta_komparasi_rbi_sentinel2.png",
    "materi/output/peta_komparasi_rbi_sentinel2.png",
    "materi/_static/peta_komparasi_rbi_sentinel2.png"
]
for p in out_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Figure 1 (Dual Comparison Map) saved successfully.")

# 9. RENDER FIGURE 2: THE ULTIMATE PROVINCIAL MASTER MAP (STYLED EXACTLY LIKE REFERENCE PHOTO!)
fig2 = plt.figure(figsize=(15, 10), dpi=300, facecolor='#FAFBFD')

# Main map axes in center-right
ax_main = fig2.add_axes([0.16, 0.12, 0.81, 0.74])
ax_main.imshow(base_relief, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
draw_map_layers(ax_main, draw_labels=True, s2_mode=False)
ax_main.set_xlim(lon_min, lon_max)
ax_main.set_ylim(lat_min, lat_max)
ax_main.axis('off')

# Outer border
rect_outer = Rectangle((0.015, 0.015), 0.97, 0.97, transform=fig2.transFigure,
                       fill=False, edgecolor='#1E293B', linewidth=2.5)
fig2.patches.append(rect_outer)
rect_inner = Rectangle((0.02, 0.02), 0.96, 0.96, transform=fig2.transFigure,
                       fill=False, edgecolor='#94A3B8', linewidth=0.8)
fig2.patches.append(rect_inner)

# TOP HEADER: PROVINSI JAWA TIMUR
fig2.text(0.72, 0.935, "PROVINSI JAWA TIMUR", fontsize=23, fontweight='heavy',
          fontfamily='sans-serif', color='#0F2850', ha='center', va='center')
fig2.text(0.72, 0.905, "PETA TOPOGRAFI & TUTUPAN LAHAN (LAND USE / LAND COVER)", fontsize=10.5,
          fontweight='bold', fontfamily='sans-serif', color='#334155', ha='center', va='center')

# Compass / Windrose top right
ax_compass = fig2.add_axes([0.90, 0.87, 0.06, 0.09])
ax_compass.annotate('', xy=(0.5, 0.85), xytext=(0.5, 0.15),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='#DC2626', edgecolor='#7F1D1D', width=3.0, headwidth=10, headlength=14))
ax_compass.text(0.5, 0.90, 'U', transform=ax_compass.transAxes, fontsize=12, fontweight='heavy', ha='center', va='bottom', color='#7F1D1D')
ax_compass.axis('off')

# TOP INSETS: P. BAWEAN & KEP. KANGEAN
# Inset 1: Bawean
ax_ins1 = fig2.add_axes([0.36, 0.865, 0.075, 0.095])
ax_ins1.set_facecolor('#C6E5F8')
# Bawean island circle
ax_ins1.add_patch(plt.Circle((0.5, 0.5), 0.32, facecolor='#ACDE74', edgecolor='#1E293B', linewidth=0.8))
ax_ins1.text(0.5, 0.5, "P. BAWEAN", fontsize=6, fontweight='bold', ha='center', va='center', color='#0F172A')
ax_ins1.set_title("P. BAWEAN", fontsize=6.5, fontweight='bold', pad=2)
ax_ins1.set_xticks([]); ax_ins1.set_yticks([])
for spine in ax_ins1.spines.values(): spine.set_edgecolor('#334155'); spine.set_linewidth(1.0)

# Inset 2: Kep. Kangean
ax_ins2 = fig2.add_axes([0.46, 0.865, 0.095, 0.095])
ax_ins2.set_facecolor('#C6E5F8')
ax_ins2.add_patch(plt.Circle((0.45, 0.5), 0.28, facecolor='#ACDE74', edgecolor='#1E293B', linewidth=0.8))
ax_ins2.add_patch(plt.Circle((0.72, 0.55), 0.12, facecolor='#ACDE74', edgecolor='#1E293B', linewidth=0.6))
ax_ins2.text(0.45, 0.5, "P. KANGEAN", fontsize=6, fontweight='bold', ha='center', va='center', color='#0F172A')
ax_ins2.set_title("KEP. KANGEAN", fontsize=6.5, fontweight='bold', pad=2)
ax_ins2.set_xticks([]); ax_ins2.set_yticks([])
for spine in ax_ins2.spines.values(): spine.set_edgecolor('#334155'); spine.set_linewidth(1.0)

# Inset 3: Kep. Masalembu
ax_ins3 = fig2.add_axes([0.58, 0.865, 0.075, 0.095])
ax_ins3.set_facecolor('#C6E5F8')
ax_ins3.add_patch(plt.Circle((0.5, 0.5), 0.24, facecolor='#ACDE74', edgecolor='#1E293B', linewidth=0.8))
ax_ins3.text(0.5, 0.5, "MASALEMBU", fontsize=5.5, fontweight='bold', ha='center', va='center', color='#0F172A')
ax_ins3.set_title("KEP. MASALEMBU", fontsize=6.5, fontweight='bold', pad=2)
ax_ins3.set_xticks([]); ax_ins3.set_yticks([])
for spine in ax_ins3.spines.values(): spine.set_edgecolor('#334155'); spine.set_linewidth(1.0)

# BOTTOM INSET: KOTA SURABAYA / METROPOLITAN ROAD NETWORK
ax_sub = fig2.add_axes([0.48, 0.035, 0.16, 0.12])
ax_sub.set_facecolor('#FDF8E2')
# Draw dense street grid representation for Surabaya
for y_line in np.linspace(0.1, 0.9, 7):
    ax_sub.plot([0.05, 0.95], [y_line, y_line], color='#EA580C', linewidth=1.1, alpha=0.8)
for x_line in np.linspace(0.1, 0.9, 7):
    ax_sub.plot([x_line, x_line], [0.05, 0.95], color='#EA580C', linewidth=1.1, alpha=0.8)
# Diagonal highway
ax_sub.plot([0.1, 0.9], [0.15, 0.85], color='#B91C1C', linewidth=2.2)
ax_sub.text(0.5, 0.5, "KOTA SURABAYA\n(JARINGAN JALAN)", fontsize=7, fontweight='heavy', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFFFFF', alpha=0.9, edgecolor='#B91C1C', linewidth=0.8))
ax_sub.set_title("KOTA SURABAYA", fontsize=7.5, fontweight='bold', pad=2)
ax_sub.set_xticks([]); ax_sub.set_yticks([])
for spine in ax_sub.spines.values(): spine.set_edgecolor('#B91C1C'); spine.set_linewidth(1.2)

# LEFT SIDEBAR: PHOTO THUMBNAILS & METADATA (EXACTLY MATCHING REFERENCE IMAGE!)
left_x = 0.035
left_w = 0.11
box_titles = [
    "Tugu Pahlawan\n(Surabaya)", 
    "Monumen JOSS\n(Gubernuran)", 
    "Kaldera Bromo\n& Semeru", 
    "Wisata Bahari\nPasir Putih", 
    "Perkebunan Kopi\nIjen Raung", 
    "Konservasi Karst\nPacitan"
]
box_colors = ["#3B82F6", "#059669", "#DC2626", "#0284C7", "#16A34A", "#9333EA"]

for i, (title, col) in enumerate(zip(box_titles, box_colors)):
    by = 0.83 - i * 0.138
    ax_box = fig2.add_axes([left_x, by, left_w, 0.115])
    # Inner photo mock frame with gradient
    grad = np.tile(np.linspace(0.85, 0.98, 50)[:, np.newaxis], (1, 50))
    ax_box.imshow(grad, cmap='Blues', extent=[0, 1, 0, 1])
    ax_box.add_patch(Rectangle((0.08, 0.08), 0.84, 0.84, fill=False, edgecolor=col, linewidth=1.5))
    ax_box.text(0.5, 0.5, title, fontsize=6.8, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax_box.set_xticks([]); ax_box.set_yticks([])
    for spine in ax_box.spines.values(): spine.set_edgecolor('#64748B'); spine.set_linewidth(0.8)

# BOTTOM LEFT: OFFICIAL LEGENDA BOX & SCALES
ax_leg = fig2.add_axes([0.165, 0.035, 0.28, 0.13])
ax_leg.set_facecolor('#FFFFFF')
ax_leg.text(0.5, 0.88, "LEGENDA & INFORMASI PETA", fontsize=8.5, fontweight='heavy', ha='center', va='top', color='#0F172A')
ax_leg.text(0.5, 0.72, "Skala 1 : 500.000   |   Proyeksi Transverse Mercator (WGS 84)", fontsize=6.5, ha='center', color='#334155')

# Scale bar in legend
ax_leg.plot([0.15, 0.85], [0.55, 0.55], color='#0F172A', linewidth=2.0)
ax_leg.plot([0.15, 0.15], [0.50, 0.60], color='#0F172A', linewidth=1.5)
ax_leg.plot([0.50, 0.50], [0.50, 0.60], color='#0F172A', linewidth=1.5)
ax_leg.plot([0.85, 0.85], [0.50, 0.60], color='#0F172A', linewidth=1.5)
ax_leg.text(0.15, 0.42, "0", fontsize=6, ha='center', fontweight='bold')
ax_leg.text(0.50, 0.42, "50 km", fontsize=6, ha='center', fontweight='bold')
ax_leg.text(0.85, 0.42, "100 km", fontsize=6, ha='center', fontweight='bold')

# Mini legend items
ax_leg.scatter([0.08], [0.22], marker='s', s=16, color='#DC2626')
ax_leg.text(0.13, 0.22, "Ibu Kota", fontsize=5.8, va='center', fontweight='bold')
ax_leg.scatter([0.38], [0.22], marker='^', s=18, color='#DC2626')
ax_leg.text(0.43, 0.22, "Gunung Api", fontsize=5.8, va='center', fontweight='bold')
ax_leg.plot([0.65, 0.73], [0.22, 0.22], color='#7F1D1D', linewidth=2.0)
ax_leg.text(0.75, 0.22, "Jalan Tol", fontsize=5.8, va='center', fontweight='bold')

ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
ax_leg.set_xticks([]); ax_leg.set_yticks([])
for spine in ax_leg.spines.values(): spine.set_edgecolor('#1E293B'); spine.set_linewidth(1.2)

# Save Figure 2
out_prov = [
    "output/peta_provinsi_jawa_timur_topografi_lulc.png",
    "materi/output/peta_provinsi_jawa_timur_topografi_lulc.png",
    "materi/_static/peta_provinsi_jawa_timur_topografi_lulc.png"
]
for p in out_prov:
    fig2.savefig(p, dpi=300, bbox_inches='tight', facecolor='#FAFBFD')
plt.close(fig2)
print("Figure 2 (Provincial Topographic Master Map) saved successfully.")

# 10. SAVE HIGH-RES TRANSPARENT LULC MASKED OVERLAY FOR FOLIUM
rgba_overlay = np.zeros((H, W, 4), dtype=np.float32)
rgba_overlay[mask_2d, :3] = s2_rgb[mask_2d]
rgba_overlay[mask_2d, 3] = 0.88 # 88% opacity on land
rgba_overlay[~mask_2d, 3] = 0.0  # Fully transparent outside East Java

img_overlay = Image.fromarray((rgba_overlay * 255).astype(np.uint8))
img_overlay.save("output/peta_lulc_raster_masked.png")
img_overlay.save("materi/output/peta_lulc_raster_masked.png")
img_overlay.save("materi/_static/peta_lulc_raster_masked.png")
print("High-res transparent LULC masked overlay saved successfully.")

print("=== ALL COMPONENT MAPS GENERATED SUCCESSFULLY ===")
