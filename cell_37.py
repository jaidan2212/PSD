import warnings
warnings.filterwarnings('ignore')
import pickle, os
import pandas as pd
import matplotlib.pyplot as plt
import folium

OUTPUT_DIR = 'output'

# === 4.1 Load Data ===
print("Memuat data Tahap 3...")
with open(f'{OUTPUT_DIR}/tahap3_clustering.pkl', 'rb') as f:
    d = pickle.load(f)

BEST_DATASET = d['BEST_DATASET']
BEST_K = d['BEST_K']
CLUSTER_COLORS = d['CLUSTER_COLORS']
df_map = d['meta_linear'] if 'Linear' in BEST_DATASET else d['meta_poly']
df_map['cluster'] = d['BEST_LABELS']

# === 4.2 & 4.3 Geocoding Cepat ===
# Hardcode koordinat agar tidak error saat menarik data lokasi
koordinat_cache = {
    'Baron Nganjuk': [-7.5083, 111.9167], 'Nunukan': [4.1333, 117.6667],
    'Sreseh, Sampang': [-7.1833, 113.2167], 'Manyar, Gresik': [-7.1500, 112.6500],
    'Kamal, Bangkalan': [-7.1833, 112.7833], 'Kedungpring Lamongan': [-7.3500, 112.2167],
    'Gresik Kota, Gresik': [-7.1556, 112.6527], 'Waru, Pamekasan': [-7.1667, 113.4833],
    'Paciran, Lamongan': [-6.8667, 112.3333], 'Kertosono, Nganjuk': [-7.5833, 112.1000],
    'Banyu Ajuh, Perumnas, Kamal': [-7.1833, 112.7833], 'Bandung Jogoroto, Jombang': [-7.5467, 112.2331],
    'Kec. Kalianget, Sumenep': [-7.0583, 113.9333], 'Jabon, Sidoarjo': [-7.5351, 112.8107],
    'Menganti, Gresik': [-7.3029, 112.5829], 'Widang, Tuban': [-7.0851, 112.1708],
    'Kwanyar, Bangkalan': [-7.1639, 112.8510], 'sambeng, lamongan': [-7.2973, 112.2709],
    'Cerme, Gresik': [-7.2243, 112.5708], 'Tikala, Manado': [1.4680, 124.8625],
    'Kerek, Tuban': [-6.8971, 111.8855], 'Wonokromo, Surabaya': [-7.3021, 112.7392],
    'Asemrowo, Surabaya': [-7.2417, 112.6888], 'Kota Sumenep, Sumenep': [-7.0067, 113.8599],
    'Socah, Bangkalan': [-7.0909, 112.7055], 'Pilangkenceng, Madiun': [-7.4996, 111.6443],
    'Tanah Merah, Bangkalan': [-7.0883, 112.8853], 'Sidoarjo, Wonoayu': [-7.4456, 112.6644],
    'Labang, Bangkalan': [-7.1396, 112.7731], 'Widodaren, Ngawi': [-7.4029, 111.2239],
    'Bangkalan, Bangkalan': [-7.0295, 112.7473], 'Warudoyong, Kota Sukabumi': [-6.9341, 106.9209],
    'Banyuajuh kamal, Bangkalan': [-7.1833, 112.7833], 'Dukun, Gresik': [-6.9964, 112.5098],
    'Kecamatan Bangkalan, Bangkalan': [-7.0295, 112.7473]
}

df_map['lat'] = df_map['daerah'].apply(lambda x: koordinat_cache.get(x, [-7.5, 112.7])[0])
df_map['lon'] = df_map['daerah'].apply(lambda x: koordinat_cache.get(x, [-7.5, 112.7])[1])

# === 4.4 Peta Interaktif (Folium) ===
print("Membuat Peta Interaktif Folium...")
m = folium.Map(location=[-2.5, 118.0], zoom_start=5)
folium.TileLayer('OpenStreetMap').add_to(m)

fg = folium.FeatureGroup(name=f"Cluster {BEST_DATASET}")
for _, row in df_map.iterrows():
    c = int(row['cluster'])
    col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
    folium.CircleMarker(
        location=[row['lat'], row['lon']], radius=12, color=col, fill=True,
        fill_opacity=0.8, tooltip=f"Cluster {c}: {row['daerah']}"
    ).add_to(fg)

fg.add_to(m)
folium.LayerControl().add_to(m)
html_path = f'{OUTPUT_DIR}/peta_clustering_interaktif.html'
m.save(html_path)
print(f"Peta HTML Interaktif tersimpan di: {html_path}")

# === 4.5 Peta Statis (Matplotlib) ===
print("Membuat Peta Statis Matplotlib...")
fig, ax = plt.subplots(figsize=(14, 8))
for c in range(BEST_K):
    mask = df_map['cluster'] == c
    sub = df_map[mask]
    ax.scatter(sub['lon'], sub['lat'], s=200, color=CLUSTER_COLORS[c],
               edgecolors='white', linewidth=1.5, label=f'Cluster {c} ({len(sub)} wil)')
    for _, row in sub.iterrows():
        ax.annotate(row['daerah'][:12], (row['lon'], row['lat']), xytext=(4,4),
                    textcoords='offset points', fontsize=7.5)

ax.set_facecolor('#E8F4FD')
ax.grid(True, linestyle='-', alpha=0.4, color='white')
ax.set_title(f'Peta Segmentasi Wilayah Polutan\nDataset: {BEST_DATASET} | k={BEST_K}', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
png_path = f'{OUTPUT_DIR}/12_peta_statis.png'
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Peta Gambar Statis tersimpan di: {png_path}")
print("\n[SELESAI] Tahap 4 Berhasil!")