import folium
from folium.raster_layers import ImageOverlay
from folium import LayerControl, FeatureGroup
import base64
import io
import os
from PIL import Image

print('Generating interactive Folium map with masked East Java raster...')

img = Image.open('materi/_static/peta_lulc_raster_masked.png')
buffered = io.BytesIO()
img.save(buffered, format='PNG')
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
data_uri = f'data:image/png;base64,{img_b64}'

lat_min, lat_max = -8.85, -6.65
lon_min, lon_max = 110.85, 114.65
bounds = [[lat_min, lon_min], [lat_max, lon_max]]
pusat = [-7.75, 112.75]

m = folium.Map(location=pusat, zoom_start=8, control_scale=True)

# Google Satellite
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(m)

# Esri Satellite
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles &copy; Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

# OpenStreetMap
folium.TileLayer(
    tiles='OpenStreetMap',
    name='OpenStreetMap',
    overlay=False,
    control=True
).add_to(m)

# Masked Raster Overlay (Daratan Jatim Saja - Persis seperti Peta Publikasi!)
ImageOverlay(
    image=data_uri,
    bounds=bounds,
    opacity=0.75,
    name='Peta Prediksi LULC Jawa Timur (Masked Raster)',
    interactive=False,
    zindex=4
).add_to(m)

# Batas AOI
grup_aoi = FeatureGroup(name='Batas Batang Studi (AOI)', show=True)
folium.Rectangle(bounds, color='yellow', weight=2, fill=False, dash_array='5').add_to(grup_aoi)
grup_aoi.add_to(m)

# Legenda Elegan
legenda_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:9999; background:rgba(255,255,255,0.92);
     padding:12px 16px; border:1.5px solid #aaa; border-radius:8px; font-size:12px; font-family:sans-serif;
     box-shadow: 0 4px 12px rgba(0,0,0,0.25); backdrop-filter: blur(4px);">
     <b style="font-size:13px; color:#111;">Klasifikasi LULC Jawa Timur (Sentinel-2A / RBI)</b><br>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#E84826;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>1. Bangunan</b> (Permukiman/Urban)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#F8E878;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>2. Sawah</b> (Lahan Pertanian)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#7DCDF5;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>3. Danau / Waduk</b> (Air Tawar)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#1C5FD7;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>4. Perairan Terbuka</b> (Laut)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#249B1E;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>5. Lahan Hijau / Hutan</b> (Vegetasi Rapat)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#96D76E;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>6. Perkebunan</b> (Vegetasi Sekunder)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#873016;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>7. Vulkanik / Tanah Terbuka</b> (Kaldera/Puncak)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#16A08C;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>8. Mangrove</b> (Pesisir Pasang Surut)</div>
     <div style="margin:4px 0"><span style="display:inline-block;width:14px;height:14px;background:#7D781E;border-radius:3px;margin-right:8px;vertical-align:middle;"></span><b>9. Tegalan / Semak Belukar</b></div>
</div>
"""
m.get_root().html.add_child(folium.Element(legenda_html))
LayerControl(collapsed=False).add_to(m)
m.fit_bounds(bounds)

m.save('materi/_static/peta_hasil_rf.html')
m.save('materi/output/peta_hasil_rf.html')
print('Successfully saved interactive Folium map to materi/_static and materi/output.')
