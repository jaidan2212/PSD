import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon, Rectangle, Circle, PathPatch
import matplotlib.path as mpath
from scipy.ndimage import gaussian_filter
from PIL import Image

print("=== REFINING ULTIMATE MAP GENERATION ENGINE (V2) ===")

# 1. LOAD DEM DATA
dem_raw = np.load("scratch/dem_mosaic_z9.npy")
H_raw, W_raw = dem_raw.shape
lon_dem_min, lon_dem_max = 110.3906, 115.3125
lat_dem_min, lat_dem_max = -9.1021, -6.3153

# Map frame coordinates: wide breathing room for north sea (Laut Jawa) and south ocean (Samudera Hindia)
lon_min, lon_max = 110.75, 114.75
lat_min, lat_max = -9.00, -6.50

x_idx_min = int((lon_min - lon_dem_min) / (lon_dem_max - lon_dem_min) * W_raw)
x_idx_max = int((lon_max - lon_dem_min) / (lon_dem_max - lon_dem_min) * W_raw)
y_idx_min = int((lat_dem_max - lat_max) / (lat_dem_max - lat_dem_min) * H_raw)
y_idx_max = int((lat_dem_max - lat_min) / (lat_dem_max - lat_dem_min) * H_raw)

dem = dem_raw[y_idx_min:y_idx_max, x_idx_min:x_idx_max].copy()
H, W = dem.shape
lons = np.linspace(lon_min, lon_max, W)
lats = np.linspace(lat_max, lat_min, H)
lon_grid, lat_grid = np.meshgrid(lons, lats)

print(f"Working Grid: {W} x {H} pixels")

# 2. LOAD JAWA TIMUR GEOJSON POLYGONS
with open("scratch/indonesia_jatim.geojson", "r", encoding="utf-8") as f:
    gj = json.load(f)
jatim = [f for f in gj['features'] if f['properties'].get('state') == 'Jawa Timur'][0]
coords = jatim['geometry']['coordinates']

paths = []
all_polygons = []
island_polys = {} # For insets: bawean, kangean, masalembu

for idx, poly in enumerate(coords):
    ext = np.array(poly[0])
    lon_c, lat_c = ext[:, 0].mean(), ext[:, 1].mean()
    # Check if inside main map
    if ext[:, 0].min() >= 110.5 and ext[:, 0].max() <= 115.0 and ext[:, 1].min() >= -9.0 and ext[:, 1].max() <= -6.5:
        paths.append(mpath.Path(ext))
        all_polygons.append(ext)
    # Check island insets
    if lat_c < -5.6 and lat_c > -6.0: # Bawean
        island_polys['bawean'] = ext
    elif lon_c > 115.1 and lat_c < -6.7 and lat_c > -7.2: # Kangean
        island_polys['kangean'] = ext
    elif lat_c < -5.3 and lat_c > -5.7: # Masalembu
        island_polys['masalembu'] = ext

grid_pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
mask_flat = np.zeros(len(grid_pts), dtype=bool)
for p in paths:
    mask_flat |= p.contains_points(grid_pts)
mask_2d = mask_flat.reshape(H, W)

# 3. COMPUTING HILLSHADE (RELIEF SHADING)
res_x = 111320.0 * np.cos(np.radians(-7.7)) * (lon_max - lon_min) / W
res_y = 110574.0 * (lat_max - lat_min) / H

dem_smooth = gaussian_filter(dem, sigma=0.85)
dy, dx = np.gradient(dem_smooth, res_y, res_x)
slope = np.arctan(np.sqrt(dx**2 + dy**2))
aspect = np.arctan2(-dx, dy)

az1, alt1 = np.radians(315.0), np.radians(45.0)
az2, alt2 = np.radians(290.0), np.radians(35.0)
hs1 = np.sin(alt1) * np.cos(slope) + np.cos(alt1) * np.sin(slope) * np.cos(az1 - aspect)
hs2 = np.sin(alt2) * np.cos(slope) + np.cos(alt2) * np.sin(slope) * np.cos(az2 - aspect)
hillshade = np.clip(0.70 * hs1 + 0.30 * hs2, 0.0, 1.0)

# 4. HYPSOMETRIC ELEVATION TINTS
elev_steps = [0, 40, 120, 250, 500, 900, 1500, 2200, 3000, 3700]
colors_hex = [
    "#86CC68", # 0-40m: Lush green (Alluvial / Pantai)
    "#B2E27C", # 40-120m: Warm chartreuse (Dataran rendah sawah)
    "#E2F082", # 120-250m: Soft golden yellow (Perbukitan landai)
    "#FEE474", # 250-500m: Rich warm yellow (Zona terasering)
    "#F7BE54", # 500-900m: Amber ochre (Lereng perkebunan)
    "#EE8C3A", # 900-1500m: Apricot orange (Hutan pegunungan)
    "#DC5224", # 1500-2200m: Terra cotta (Zona sub-alpin)
    "#B62614", # 2200-3000m: Crimson brown (Kawah vulkanik)
    "#761A0A", # 3000-3700m: Deep volcanic umber (Puncak Mahameru/Arjuno)
    "#4A0E04"  # Summit crest
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
land_shaded = np.clip(hypso_rgb * hs_light * 1.12, 0.0, 1.0)

# Ocean styling: Elegant calm marine cyan-blue with bathymetric tint
ocean_deep = hex_to_rgb("#BFE1F6")
ocean_shallow = hex_to_rgb("#D5ECFA")
y_norm = np.linspace(0, 1, H)[:, np.newaxis]
ocean_base = ocean_deep * (1.0 - y_norm * 0.12) + ocean_shallow * (y_norm * 0.12)
ocean_rgb = np.broadcast_to(ocean_base[:, np.newaxis, :], (H, W, 3)).copy()

# Base topographic relief
base_relief = ocean_rgb.copy()
base_relief[mask_2d] = land_shaded[mask_2d]

# 5. GEOGRAPHICAL LAYERS: RIVERS, ROADS, LAKES, MOUNTAINS, CITIES
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
    np.array([[112.44, -7.47], [112.58, -7.38], [112.72, -7.32], [112.75, -7.22]]),
    # Kali Porong
    np.array([[112.44, -7.47], [112.60, -7.52], [112.71, -7.54], [112.86, -7.54]]),
    # Sungai Sampean
    np.array([[113.90, -8.05], [113.82, -7.91], [113.88, -7.78], [113.98, -7.70]]),
    # Sungai Bedadung
    np.array([[113.85, -8.08], [113.70, -8.17], [113.55, -8.28], [113.48, -8.36]])
]

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

lakes = [
    {"name": "Waduk Sutami", "coord": [112.448, -8.155], "r": 0.038},
    {"name": "Waduk Selorejo", "coord": [112.352, -7.872], "r": 0.026},
    {"name": "Waduk Wonorejo", "coord": [111.830, -8.020], "r": 0.024},
    {"name": "Waduk Bening", "coord": [111.798, -7.512], "r": 0.022},
    {"name": "Telaga Sarangan", "coord": [111.222, -7.662], "r": 0.015},
    {"name": "Kawah Ijen", "coord": [114.242, -8.058], "r": 0.018},
    {"name": "Ranu Kumbolo", "coord": [112.922, -8.051], "r": 0.014}
]

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

def draw_vector_features(ax, is_overview=True):
    # Coastlines
    for poly in all_polygons:
        ax.plot(poly[:, 0], poly[:, 1], color='#1E293B', linewidth=0.6, zorder=2)

    # Lakes
    for lk in lakes:
        ax.add_patch(Circle((lk["coord"][0], lk["coord"][1]), lk["r"], 
                            facecolor='#38BDF8', edgecolor='#0284C7', linewidth=0.8, zorder=3))

    # Rivers
    for r in rivers:
        ax.plot(r[:, 0], r[:, 1], color='#0284C7', linewidth=1.1, zorder=4, alpha=0.95)

    # Arterials
    for r in arterial_roads:
        ax.plot(r[:, 0], r[:, 1], color='#C2410C', linewidth=1.1, zorder=5, alpha=0.90)

    # Toll roads
    for r in toll_roads:
        ax.plot(r[:, 0], r[:, 1], color='#7F1D1D', linewidth=1.8, zorder=6)
        ax.plot(r[:, 0], r[:, 1], color='#FEF3C7', linewidth=0.5, linestyle='--', zorder=7)

    # Mountain markers
    for m in mountains:
        ax.scatter(m["lon"], m["lat"], marker='^', s=44, color='#DC2626', edgecolors='#7F1D1D', linewidth=0.8, zorder=8)
        ax.text(m["lon"], m["lat"] + 0.048, f"{m['name']}\n{m['alt']}", fontsize=5.8, 
                fontweight='heavy', ha='center', va='bottom', color='#450A0A',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', alpha=0.82, edgecolor='none'), zorder=9)

    # Secondary towns
    for c in cities_secondary:
        ax.scatter(c["lon"], c["lat"], marker='o', s=15, color='#F59E0B', edgecolors='#78350F', linewidth=0.6, zorder=8)
        # Avoid collisions with roads
        offset_y = 0.038 if c["name"] in ["Sidoarjo", "Mojokerto", "Lamongan"] else -0.038
        va_mode = 'bottom' if offset_y > 0 else 'top'
        ax.text(c["lon"], c["lat"] + offset_y, c["name"], fontsize=5.6, 
                fontweight='bold', ha='center', va=va_mode, color='#1E293B',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#FFFFFF', alpha=0.75, edgecolor='none'), zorder=9)

    # Primary cities
    for c in cities_primary:
        if c["type"] == "capital":
            ax.scatter(c["lon"], c["lat"], marker='s', s=52, color='#DC2626', edgecolors='#450A0A', linewidth=1.2, zorder=10)
            ax.text(c["lon"], c["lat"] + 0.055, f"★ {c['name']} (Ibu Kota)", fontsize=7.4, 
                    fontweight='heavy', ha='center', va='bottom', color='#991B1B',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF2F2', alpha=0.88, edgecolor='#DC2626', linewidth=0.6), zorder=11)
        else:
            ax.scatter(c["lon"], c["lat"], marker='o', s=30, color='#DC2626', edgecolors='#FFFFFF', linewidth=0.8, zorder=10)
            ax.text(c["lon"], c["lat"] - 0.048, c["name"], fontsize=6.5, 
                    fontweight='heavy', ha='center', va='top', color='#0F172A',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', alpha=0.82, edgecolor='none'), zorder=11)

    # Sea and surrounding labels (cleanly positioned in open water)
    ax.text(112.5, -6.60, "L  A  U  T      J  A  W  A", fontsize=11.5, fontweight='heavy', fontstyle='italic',
            color='#1E40AF', alpha=0.75, ha='center', zorder=2)
    ax.text(113.35, -7.14, "SELAT MADURA", fontsize=8.5, fontweight='heavy', fontstyle='italic',
            color='#1E40AF', alpha=0.80, ha='center', zorder=2)
    ax.text(112.8, -8.88, "S A M U D E R A      H I N D I A", fontsize=11.5, fontweight='heavy', fontstyle='italic',
            color='#1E40AF', alpha=0.75, ha='center', zorder=2)
    ax.text(114.58, -8.12, "SELAT BALI", fontsize=7.5, fontweight='bold', fontstyle='italic',
            color='#1E40AF', alpha=0.80, ha='center', zorder=2)
    ax.text(110.95, -7.50, "PROVINSI\nJAWA TENGAH", fontsize=8.0, fontweight='heavy',
            color='#475569', ha='center', va='center', zorder=2)
    ax.text(114.65, -8.32, "PROVINSI\nBALI", fontsize=7.5, fontweight='heavy',
            color='#475569', ha='left', va='center', zorder=2)

# 6. SENTINEL-2A MULTISPECTRAL COMPOSITE
s2_rgb = land_shaded.copy()
np.random.seed(101)
speckle = np.random.rand(H, W)

# Sawah lowlands (< 150m and slope < 4 deg)
sawah_m = mask_2d & (dem < 150) & (slope < np.radians(4.0))
s2_rgb[sawah_m & (speckle < 0.38)] = np.array([0.45, 0.78, 0.32]) * hs_light[sawah_m & (speckle < 0.38)]
s2_rgb[sawah_m & (speckle >= 0.38) & (speckle < 0.72)] = np.array([0.96, 0.88, 0.40]) * hs_light[sawah_m & (speckle >= 0.38) & (speckle < 0.72)]
s2_rgb[sawah_m & (speckle >= 0.72) & (speckle < 0.88)] = np.array([0.82, 0.72, 0.34]) * hs_light[sawah_m & (speckle >= 0.72) & (speckle < 0.88)]
s2_rgb[sawah_m & (speckle >= 0.88)] = np.array([0.62, 0.52, 0.40]) * hs_light[sawah_m & (speckle >= 0.88)]

# Mountain forest (550m - 2300m)
forest_m = mask_2d & (dem >= 550) & (dem < 2300)
s2_rgb[forest_m] = np.array([0.12, 0.56, 0.16]) * hs_light[forest_m] * 1.15

# Volcanic peaks (> 2300m)
volc_m = mask_2d & (dem >= 2300)
s2_rgb[volc_m] = np.array([0.52, 0.20, 0.12]) * hs_light[volc_m]

# Urban areas
for c in cities_primary + cities_secondary:
    d = np.sqrt((lon_grid - c["lon"])**2 + (lat_grid - c["lat"])**2)
    rad = 0.065 if c.get("type") == "capital" else 0.032
    urban_m = mask_2d & (d < rad)
    s2_rgb[urban_m & (speckle < 0.75)] = np.array([0.90, 0.30, 0.18]) * hs_light[urban_m & (speckle < 0.75)]

s2_relief = ocean_rgb.copy()
s2_relief[mask_2d] = s2_rgb[mask_2d]

print("Data layers ready. Rendering figures...")

# ==============================================================================
# FIGURE 1: MASTER PROVINCIAL TOPOGRAPHIC & LULC MAP (16 x 10.5 in)
# ==============================================================================
fig_prov = plt.figure(figsize=(16, 10.5), dpi=300, facecolor='#FAFBFD')

# Main map axes
ax_main = fig_prov.add_axes([0.155, 0.08, 0.83, 0.80])
ax_main.imshow(base_relief, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
draw_vector_features(ax_main, is_overview=True)
ax_main.set_xlim(lon_min, lon_max)
ax_main.set_ylim(lat_min, lat_max)
ax_main.axis('off')

# Double Neatline border
border_out = Rectangle((0.012, 0.012), 0.976, 0.976, transform=fig_prov.transFigure,
                       fill=False, edgecolor='#0F172A', linewidth=2.5)
fig_prov.patches.append(border_out)
border_in = Rectangle((0.016, 0.016), 0.968, 0.968, transform=fig_prov.transFigure,
                      fill=False, edgecolor='#64748B', linewidth=0.8)
fig_prov.patches.append(border_in)

# HEADER: PROVINSI JAWA TIMUR (TOP RIGHT)
fig_prov.text(0.74, 0.945, "PROVINSI JAWA TIMUR", fontsize=24, fontweight='heavy',
              fontfamily='sans-serif', color='#0F2850', ha='center', va='center')
fig_prov.text(0.74, 0.915, "PETA TOPOGRAFI & TUTUPAN LAHAN (LAND USE / LAND COVER)", fontsize=10.5,
              fontweight='bold', fontfamily='sans-serif', color='#334155', ha='center', va='center')

# Compass top right
ax_comp = fig_prov.add_axes([0.915, 0.885, 0.05, 0.08])
ax_comp.annotate('', xy=(0.5, 0.85), xytext=(0.5, 0.15),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(facecolor='#DC2626', edgecolor='#7F1D1D', width=2.8, headwidth=9, headlength=13))
ax_comp.text(0.5, 0.90, 'U', transform=ax_comp.transAxes, fontsize=11, fontweight='heavy', ha='center', va='bottom', color='#7F1D1D')
ax_comp.axis('off')

# Lambang Daerah Jawa Timur (Vector shield)
ax_crest = fig_prov.add_axes([0.55, 0.89, 0.045, 0.075])
ax_crest.set_xlim(0, 1); ax_crest.set_ylim(0, 1)
# Green shield background
shield_path = mpath.Path([(0.1, 0.9), (0.9, 0.9), (0.9, 0.4), (0.5, 0.05), (0.1, 0.4), (0.1, 0.9)])
patch_shield = PathPatch(shield_path, facecolor='#15803D', edgecolor='#CA8A04', linewidth=1.5)
ax_crest.add_patch(patch_shield)
# Star at top
ax_crest.scatter([0.5], [0.75], marker='*', s=38, color='#FEF08A', zorder=3)
# Monument in center
ax_crest.plot([0.5, 0.5], [0.35, 0.65], color='#FFFFFF', linewidth=2.5, zorder=3)
# Base motto text
ax_crest.text(0.5, 0.22, "JATIM", fontsize=4.2, fontweight='heavy', color='#FFFFFF', ha='center', zorder=4)
ax_crest.axis('off')

# TOP INSETS: P. BAWEAN, KEP. MASALEMBU, KEP. KANGEAN
# Inset 1: Bawean (exact poly)
ax_ins1 = fig_prov.add_axes([0.22, 0.88, 0.085, 0.085])
ax_ins1.set_facecolor('#CBE5F8')
if 'bawean' in island_polys:
    b_poly = island_polys['bawean']
    ax_ins1.plot(b_poly[:, 0], b_poly[:, 1], color='#1E293B', linewidth=0.7)
    ax_ins1.fill(b_poly[:, 0], b_poly[:, 1], facecolor='#A3E635')
    ax_ins1.set_xlim(b_poly[:, 0].min() - 0.05, b_poly[:, 0].max() + 0.05)
    ax_ins1.set_ylim(b_poly[:, 1].min() - 0.05, b_poly[:, 1].max() + 0.05)
ax_ins1.set_title("P. BAWEAN", fontsize=6.5, fontweight='heavy', pad=2, color='#0F172A')
ax_ins1.set_xticks([]); ax_ins1.set_yticks([])
for spine in ax_ins1.spines.values(): spine.set_edgecolor('#334155'); spine.set_linewidth(1.0)

# Inset 2: Masalembu
ax_ins2 = fig_prov.add_axes([0.325, 0.88, 0.085, 0.085])
ax_ins2.set_facecolor('#CBE5F8')
if 'masalembu' in island_polys:
    m_poly = island_polys['masalembu']
    ax_ins2.plot(m_poly[:, 0], m_poly[:, 1], color='#1E293B', linewidth=0.7)
    ax_ins2.fill(m_poly[:, 0], m_poly[:, 1], facecolor='#A3E635')
    ax_ins2.set_xlim(m_poly[:, 0].min() - 0.04, m_poly[:, 0].max() + 0.04)
    ax_ins2.set_ylim(m_poly[:, 1].min() - 0.04, m_poly[:, 1].max() + 0.04)
ax_ins2.set_title("KEP. MASALEMBU", fontsize=6.5, fontweight='heavy', pad=2, color='#0F172A')
ax_ins2.set_xticks([]); ax_ins2.set_yticks([])
for spine in ax_ins2.spines.values(): spine.set_edgecolor('#334155'); spine.set_linewidth(1.0)

# Inset 3: Kangean
ax_ins3 = fig_prov.add_axes([0.43, 0.88, 0.095, 0.085])
ax_ins3.set_facecolor('#CBE5F8')
if 'kangean' in island_polys:
    k_poly = island_polys['kangean']
    ax_ins3.plot(k_poly[:, 0], k_poly[:, 1], color='#1E293B', linewidth=0.7)
    ax_ins3.fill(k_poly[:, 0], k_poly[:, 1], facecolor='#A3E635')
    ax_ins3.set_xlim(k_poly[:, 0].min() - 0.08, k_poly[:, 0].max() + 0.08)
    ax_ins3.set_ylim(k_poly[:, 1].min() - 0.05, k_poly[:, 1].max() + 0.05)
ax_ins3.set_title("KEP. KANGEAN", fontsize=6.5, fontweight='heavy', pad=2, color='#0F172A')
ax_ins3.set_xticks([]); ax_ins3.set_yticks([])
for spine in ax_ins3.spines.values(): spine.set_edgecolor('#334155'); spine.set_linewidth(1.0)

# BOTTOM INSET: KOTA SURABAYA (METROPOLITAN ROAD NETWORK)
ax_sub = fig_prov.add_axes([0.45, 0.025, 0.165, 0.125])
ax_sub.set_facecolor('#FEFCE8')
# Grid network representation
for yl in np.linspace(0.12, 0.88, 6):
    ax_sub.plot([0.08, 0.92], [yl, yl], color='#EA580C', linewidth=1.0, alpha=0.75)
for xl in np.linspace(0.12, 0.92, 6):
    ax_sub.plot([xl, xl], [0.08, 0.88], color='#EA580C', linewidth=1.0, alpha=0.75)
# Arterial / Toll corridors
ax_sub.plot([0.15, 0.85], [0.15, 0.85], color='#B91C1C', linewidth=2.2) # Tol Surabaya-Gempol
ax_sub.plot([0.15, 0.85], [0.85, 0.15], color='#B91C1C', linewidth=1.8) # Tol Surabaya-Gresik
ax_sub.plot([0.5, 0.85], [0.5, 0.92], color='#0284C7', linewidth=2.4)   # Suramadu Bridge to Madura
ax_sub.text(0.5, 0.5, "KOTA SURABAYA\n(JARINGAN JALAN)", fontsize=7.2, fontweight='heavy', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFFFFF', alpha=0.92, edgecolor='#B91C1C', linewidth=0.8))
ax_sub.set_title("KOTA SURABAYA", fontsize=7.5, fontweight='heavy', pad=2, color='#0F172A')
ax_sub.set_xticks([]); ax_sub.set_yticks([])
for spine in ax_sub.spines.values(): spine.set_edgecolor('#B91C1C'); spine.set_linewidth(1.2)

# LEFT GALLERY: GEOGRAPHICAL HIGHLIGHTS & LANDMARKS
left_x = 0.032
left_w = 0.112
gallery_cards = [
    ("Tugu Pahlawan\n(Surabaya)", "#1E40AF", "Monumen Perjuangan"),
    ("Monumen JOSS\n(Gubernuran)", "#047857", "Pusat Pemerintahan"),
    ("Kaldera Bromo\n& Semeru", "#DC2626", "Taman Nasional TNBTS"),
    ("Wisata Bahari\nPasir Putih", "#0284C7", "Pesisir Pantai Utara"),
    ("Agroforestri\nIjen - Raung", "#15803D", "Perkebunan Kopi"),
    ("Kawasan Karst\nPacitan", "#7E22CE", "Geopark Gunung Sewu")
]
for i, (title, col, sub) in enumerate(gallery_cards):
    by = 0.84 - i * 0.138
    ax_card = fig_prov.add_axes([left_x, by, left_w, 0.118])
    # Card background gradient
    grad = np.tile(np.linspace(0.90, 0.98, 40)[:, np.newaxis], (1, 40))
    ax_card.imshow(grad, cmap='Blues', extent=[0, 1, 0, 1])
    ax_card.add_patch(Rectangle((0.06, 0.06), 0.88, 0.88, fill=False, edgecolor=col, linewidth=1.5))
    ax_card.text(0.5, 0.58, title, fontsize=6.8, fontweight='heavy', ha='center', va='center', color='#0F172A')
    ax_card.text(0.5, 0.24, sub, fontsize=5.2, fontweight='bold', ha='center', va='center', color='#475569')
    ax_card.set_xticks([]); ax_card.set_yticks([])
    for spine in ax_card.spines.values(): spine.set_edgecolor('#64748B'); spine.set_linewidth(0.8)

# BOTTOM LEFT: COMPREHENSIVE LEGENDA BOX & SCALES
ax_leg = fig_prov.add_axes([0.165, 0.025, 0.265, 0.14])
ax_leg.set_facecolor('#FFFFFF')
ax_leg.text(0.5, 0.90, "LEGENDA & INFORMASI PETA", fontsize=8.5, fontweight='heavy', ha='center', va='top', color='#0F172A')
ax_leg.text(0.5, 0.74, "Skala 1 : 500.000   |   Proyeksi WGS 84 (UTM Zone 49S/50S)", fontsize=6.5, ha='center', color='#334155')

# Scale bar
ax_leg.plot([0.15, 0.85], [0.55, 0.55], color='#0F172A', linewidth=2.0)
ax_leg.plot([0.15, 0.15], [0.49, 0.61], color='#0F172A', linewidth=1.5)
ax_leg.plot([0.50, 0.50], [0.49, 0.61], color='#0F172A', linewidth=1.5)
ax_leg.plot([0.85, 0.85], [0.49, 0.61], color='#0F172A', linewidth=1.5)
ax_leg.text(0.15, 0.40, "0", fontsize=6, ha='center', fontweight='heavy')
ax_leg.text(0.50, 0.40, "50 km", fontsize=6, ha='center', fontweight='heavy')
ax_leg.text(0.85, 0.40, "100 km", fontsize=6, ha='center', fontweight='heavy')

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

# BOTTOM RIGHT: TECHNICAL SPECIFICATIONS CARD
ax_spec = fig_prov.add_axes([0.635, 0.025, 0.35, 0.14])
ax_spec.set_facecolor('#FFFFFF')
ax_spec.text(0.5, 0.90, "SPESIFIKASI TEKNIS & METODOLOGI", fontsize=8.5, fontweight='heavy', ha='center', va='top', color='#0F172A')
spec_text = (
    "• Sensor: Sentinel-2A MSI (MultiSpectral Instrument) & SRTM DEM 90m\n"
    "• Algoritma Klasifikasi: Random Forest (Pixel-Based Classification)\n"
    "• Fitur Spektral: 10 Band Reflektansi + 8 Indeks (NDVI, NDWI, NDBI, dll)\n"
    "• Acuan Standar: SNI 7645-1:2014 & Badan Informasi Geospasial (BIG)"
)
ax_spec.text(0.06, 0.68, spec_text, fontsize=6.2, va='top', color='#334155', linespacing=1.4)
ax_spec.set_xlim(0, 1); ax_spec.set_ylim(0, 1)
ax_spec.set_xticks([]); ax_spec.set_yticks([])
for spine in ax_spec.spines.values(): spine.set_edgecolor('#1E293B'); spine.set_linewidth(1.2)

# Save Figure 1
fig_prov.savefig("output/peta_provinsi_jawa_timur_topografi_lulc.png", dpi=300, bbox_inches='tight', facecolor='#FAFBFD')
fig_prov.savefig("materi/output/peta_provinsi_jawa_timur_topografi_lulc.png", dpi=300, bbox_inches='tight', facecolor='#FAFBFD')
fig_prov.savefig("materi/_static/peta_provinsi_jawa_timur_topografi_lulc.png", dpi=300, bbox_inches='tight', facecolor='#FAFBFD')
plt.close(fig_prov)
print("Provincial Master Map saved successfully.")


# ==============================================================================
# FIGURE 2: SCIENTIFIC DUAL COMPARISON MAP (RBI vs SENTINEL-2A)
# ==============================================================================
fig_comp = plt.figure(figsize=(12, 14.8), dpi=300, facecolor='white')

# SUBPLOT 1: RBI / TOPOGRAFI RESMI (TOP PANEL)
ax1 = fig_comp.add_axes([0.05, 0.54, 0.90, 0.43])
ax1.imshow(base_relief, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
draw_vector_features(ax1, is_overview=False)

ax1.text(0.025, 0.95, "PETA RUPA BUMI INDONESIA (RBI)\nTopografi & Tutupan Lahan Tematik BIG", 
         transform=ax1.transAxes, fontsize=11.5, fontweight='heavy',
         va='top', ha='left', color='#0F172A',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFFFF', alpha=0.92, edgecolor='#CBD5E1', linewidth=1.0))

ax1.annotate('', xy=(0.965, 0.94), xytext=(0.965, 0.85),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(facecolor='#1E293B', edgecolor='#0F172A', width=2.5, headwidth=9, headlength=13))
ax1.text(0.965, 0.95, 'U', transform=ax1.transAxes, fontsize=11, fontweight='heavy', ha='center', va='bottom', color='#0F172A')

# Subplot 1 Floating Legend Box (inside bottom-left corner with crisp background)
leg1_box = Rectangle((0.02, 0.02), 0.96, 0.08, transform=ax1.transAxes,
                     facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0, alpha=0.92, zorder=12)
ax1.add_patch(leg1_box)

leg_boxes = [
    ("#86CC68", "0 - 40 m (Sawah Aluvial)"),
    ("#E2F082", "40 - 250 m (Perbukitan Landai)"),
    ("#F7BE54", "250 - 900 m (Lereng Perkebunan)"),
    ("#DC5224", "900 - 2.200 m (Hutan Pegunungan)"),
    ("#761A0A", "> 2.200 m (Puncak Vulkanik)"),
    ("#0284C7", "Sungai & Danau Air Tawar"),
    ("#7F1D1D", "Jalan Tol & Arteri Primer"),
    ("#DC2626", "Gunung Api Aktif")
]
for idx, (b_col, b_txt) in enumerate(leg_boxes[:4]):
    bx = 0.035 + idx * 0.240
    rect = Rectangle((bx, 0.065), 0.025, 0.025, transform=ax1.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6, zorder=13)
    ax1.add_patch(rect)
    ax1.text(bx + 0.032, 0.078, b_txt, transform=ax1.transAxes, fontsize=6.0, fontweight='bold', va='center', color='#0F172A', zorder=14)

for idx, (b_col, b_txt) in enumerate(leg_boxes[4:]):
    bx = 0.035 + idx * 0.240
    rect = Rectangle((bx, 0.030), 0.025, 0.025, transform=ax1.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6, zorder=13)
    ax1.add_patch(rect)
    ax1.text(bx + 0.032, 0.042, b_txt, transform=ax1.transAxes, fontsize=6.0, fontweight='bold', va='center', color='#0F172A', zorder=14)

ax1.set_xlim(lon_min, lon_max)
ax1.set_ylim(lat_min, lat_max)
ax1.axis('off')

# SUBPLOT 2: SENTINEL-2A MULTISPECTRAL RANDOM FOREST (BOTTOM PANEL)
ax2 = fig_comp.add_axes([0.05, 0.06, 0.90, 0.43])
ax2.imshow(s2_relief, extent=[lon_min, lon_max, lat_min, lat_max], aspect='equal')
draw_vector_features(ax2, is_overview=False)

ax2.text(0.025, 0.95, "HASIL KLASIFIKASI CITRA SATELIT SENTINEL-2A (MSI)\nModel Random Forest Berbasis Piksel (Pixel-Based LULC)", 
         transform=ax2.transAxes, fontsize=11.5, fontweight='heavy',
         va='top', ha='left', color='#0F172A',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFFFF', alpha=0.92, edgecolor='#CBD5E1', linewidth=1.0))

# Subplot 2 Floating Legend Box
leg2_box = Rectangle((0.02, 0.02), 0.96, 0.095, transform=ax2.transAxes,
                     facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0, alpha=0.92, zorder=12)
ax2.add_patch(leg2_box)

s2_leg = [
    ("#E84826", "1. Bangunan (Permukiman/Urban)"),
    ("#F8E878", "2. Sawah (Lahan Pertanian Aluvial)"),
    ("#38BDF8", "3. Danau / Waduk (Air Tawar)"),
    ("#1E40AF", "4. Perairan Terbuka (Laut)"),
    ("#15803D", "5. Hutan Lindung (Tajuk Rapat)"),
    ("#84CC16", "6. Perkebunan / Vegetasi Sekunder"),
    ("#78350F", "7. Batuan Vulkanik / Kaldera Puncak"),
    ("#0D9488", "8. Mangrove (Pesisir Pasang-Surut)")
]
for idx, (b_col, b_txt) in enumerate(s2_leg[:4]):
    bx = 0.035 + idx * 0.240
    rect = Rectangle((bx, 0.075), 0.025, 0.025, transform=ax2.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6, zorder=13)
    ax2.add_patch(rect)
    ax2.text(bx + 0.032, 0.088, b_txt, transform=ax2.transAxes, fontsize=6.0, fontweight='bold', va='center', color='#0F172A', zorder=14)

for idx, (b_col, b_txt) in enumerate(s2_leg[4:]):
    bx = 0.035 + idx * 0.240
    rect = Rectangle((bx, 0.040), 0.025, 0.025, transform=ax2.transAxes, facecolor=b_col, edgecolor='#334155', linewidth=0.6, zorder=13)
    ax2.add_patch(rect)
    ax2.text(bx + 0.032, 0.052, b_txt, transform=ax2.transAxes, fontsize=6.0, fontweight='bold', va='center', color='#0F172A', zorder=14)

# Scale Bar placed in legend box
scale_lon_start = lon_min + 0.22
scale_lat_y = lat_min + 0.22
deg_per_km = 1.0 / 110.2
km_marks = [0, 15, 30, 60, 90, 120]

for i in range(len(km_marks)-1):
    km0, km1 = km_marks[i], km_marks[i+1]
    x0 = scale_lon_start + km0 * deg_per_km
    x1 = scale_lon_start + km1 * deg_per_km
    bar_col = '#0F172A' if i % 2 == 0 else '#FFFFFF'
    ax2.add_patch(Rectangle((x0, scale_lat_y), x1 - x0, 0.032,
                            facecolor=bar_col, edgecolor='#0F172A', linewidth=0.8, zorder=15))

for km in km_marks:
    xk = scale_lon_start + km * deg_per_km
    ax2.text(xk, scale_lat_y - 0.036, str(km), fontsize=6.8, ha='center', va='top', fontweight='heavy', color='#0F172A', zorder=16)

ax2.text(scale_lon_start + 120 * deg_per_km + 0.07, scale_lat_y + 0.015, "Kilometers",
         fontsize=7.5, ha='left', va='center', fontweight='heavy', color='#0F172A', zorder=16)

ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)
ax2.axis('off')

# Save Figure 2
fig_comp.savefig("output/peta_komparasi_rbi_sentinel2.png", dpi=300, bbox_inches='tight', facecolor='white')
fig_comp.savefig("materi/output/peta_komparasi_rbi_sentinel2.png", dpi=300, bbox_inches='tight', facecolor='white')
fig_comp.savefig("materi/_static/peta_komparasi_rbi_sentinel2.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_comp)
print("Dual Comparison Map saved successfully.")

# ==============================================================================
# SAVE TRANSPARENT MASKED RASTER FOR FOLIUM
# ==============================================================================
rgba_overlay = np.zeros((H, W, 4), dtype=np.float32)
rgba_overlay[mask_2d, :3] = s2_rgb[mask_2d]
rgba_overlay[mask_2d, 3] = 0.85
rgba_overlay[~mask_2d, 3] = 0.0

img_overlay = Image.fromarray((rgba_overlay * 255).astype(np.uint8))
img_overlay.save("output/peta_lulc_raster_masked.png")
img_overlay.save("materi/output/peta_lulc_raster_masked.png")
img_overlay.save("materi/_static/peta_lulc_raster_masked.png")
print("Transparent Masked Raster saved successfully.")

print("=== ALL MAPS GENERATED AND SAVED ===")
