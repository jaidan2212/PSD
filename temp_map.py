import pickle, os
import pandas as pd
import folium
from folium.plugins import MiniMap
import requests

# Load data from step 3
OUTPUT_DIR = 'materi/output'
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = 'output'
with open(f'{OUTPUT_DIR}/tahap3_clustering.pkl', 'rb') as f:
    d = pickle.load(f)

BEST_DATASET = d['BEST_DATASET']
CLUSTER_COLORS = d['CLUSTER_COLORS']
BEST_PER_DIM = d['BEST_PER_DIM']

# Geocoding hardcode
koordinat_cache = {
    'Baron Nganjuk': [-7.5083, 111.9167], 'Nunukan': [4.1333, 117.6667],
    'Sreseh, Sampang': [-7.1833, 113.2167], 'Manyar, Gresik': [-7.1500, 112.6500],
    'Kamal, Bangkalan': [-7.1833, 112.7833], 'Kedungpring Lamongan': [-7.3500, 112.2167],
    'Gresik Kota, Gresik': [-7.1556, 112.6527], 'Waru, Pamekasan': [-7.1667, 113.4833],
    'Paciran, Lamongan': [-6.8667, 112.3333], 'Kertosono, Nganjuk': [-7.5833, 112.1000],
    'Banyu Ajuh, Perumnas, Kamal': [-7.1833, 112.7833], 'Bandung - Jogoroto, Jombang': [-7.5467, 112.2331],
    'Bandung Jogoroto, Jombang': [-7.5467, 112.2331],
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
    'Kecamatan Bangkalan, Bangkalan': [-7.0295, 112.7473],
    'Kecamatan Bangkalan,Bangkalan': [-7.0295, 112.7473]
}

m = folium.Map(
    location=[-7.2, 112.7], 
    zoom_start=8, 
    tiles='OpenStreetMap'
)

minimap = MiniMap(
    toggle_display=True,
    tile_layer='OpenStreetMap'
)
m.add_child(minimap)

# Fetch GeoJSON
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-edit.geojson"
try:
    geojson_data = requests.get(geojson_url).json()
except Exception as e:
    print(f"Error loading GeoJSON: {e}")
    geojson_data = None

# Add feature groups for each configuration
for key, info in BEST_PER_DIM.items():
    dataset_name = info['dataset']
    sil = info['silhouette']
    k = info['k']
    
    # Layer name exactly as in screenshot
    metode, dimensi = dataset_name.split('-')
    layer_name = f'{metode} {dimensi} dimensi | k={k}, sil={sil:.3f}'
    
    fg = folium.FeatureGroup(name=layer_name, show=(dataset_name == BEST_DATASET))
    
    df_map = d['meta_linear'] if 'Linear' in dataset_name else d['meta_poly']
    df_map['cluster'] = info['labels']
    df_map['lat'] = df_map['daerah'].apply(lambda x: koordinat_cache.get(x, [-7.5, 112.7])[0])
    df_map['lon'] = df_map['daerah'].apply(lambda x: koordinat_cache.get(x, [-7.5, 112.7])[1])
    
    if geojson_data:
        def style_function(feature, df=df_map):
            name = feature['properties'].get('name', '').lower()
            if not name:
                return {'fillColor': 'none', 'color': 'grey', 'weight': 0.5, 'fillOpacity': 0}
            
            # find if any daerah in df_map matches this name
            # name in geojson might be "Gresik", but in df_map it's "Menganti, Gresik"
            # so we check if the geojson name is in the daerah name
            matches = df[df['daerah'].str.lower().str.contains(name, na=False)]
            if len(matches) > 0:
                # Mode of the cluster (most common cluster in that regency)
                c = int(matches.iloc[0]['cluster'])
                col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                return {'fillColor': col, 'color': 'black', 'weight': 1.5, 'fillOpacity': 0.5}
            else:
                return {'fillColor': 'none', 'color': 'grey', 'weight': 0.5, 'fillOpacity': 0}
        
        folium.GeoJson(
            geojson_data,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Wilayah:'])
        ).add_to(fg)

    for _, row in df_map.iterrows():
        c = int(row['cluster'])
        col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']], radius=14, color=col, fill=True,
            fill_color=col, fill_opacity=0.9, tooltip=f"Cluster {c}: {row['daerah']}"
        ).add_to(fg)
        
        # add text C0, C1 inside marker
        folium.map.Marker(
            [row['lat'], row['lon']],
            icon=folium.DivIcon(html=f'<div style="font-size: 8pt; font-weight: bold; color: white; text-align: center; width: 28px; line-height: 28px; margin-left:-14px; margin-top:-14px;">C{c}</div>')
        ).add_to(fg)
    
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Add the legend box
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 340px; height: auto; 
     background-color: white; z-index:9999; font-size:12px;
     border:2px solid grey; border-radius:10px; padding: 10px;
     box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
     <b>Cluster Terbaik per Dimensi</b><br>
     <span style="color:grey; font-size:11px;">Pilih layer di kanan atas (★ = terbaik global)</span><br><br>
     <table style="width:100%; border-collapse: collapse;">
       <tr style="border-bottom: 1px solid #ddd; text-align: left;">
         <th>Dataset</th><th>k</th><th>Sil.</th><th>Ukuran cluster</th>
       </tr>'''

for key, info in BEST_PER_DIM.items():
    dataset_name = info['dataset']
    sil = info['silhouette']
    k = info['k']
    ukuran = ' / '.join(map(str, info['ukuran']))
    star = ' ★' if dataset_name == BEST_DATASET else ''
    weight = 'bold' if dataset_name == BEST_DATASET else 'normal'
    legend_html += f'<tr style="border-bottom: 1px solid #eee;"><td style="font-weight:{weight};">{dataset_name}{star}</td><td>k={k}</td><td>{sil:.3f}</td><td>{ukuran}</td></tr>'

legend_html += '</table><br>'

for i in range(5): # Show max 5 clusters in legend
    col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    legend_html += f'<span style="color:{col}; font-size:16px;">⬤</span> C{i}&nbsp;&nbsp;&nbsp;'

legend_html += '</div>'

m.get_root().html.add_child(folium.Element(legend_html))

m.save('materi/_static/peta_clustering_interaktif.html')
print("Map generated successfully!")
