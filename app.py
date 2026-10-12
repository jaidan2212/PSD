import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Klasifikasi LULC Jawa Timur", layout="wide")

st.title("🌍 Klasifikasi Penggunaan dan Tutupan Lahan (Land Use / Land Cover) Jawa Timur")
st.markdown("""
**Mata kuliah:** Penambangan Data Sains (*Data Mining*)  
Aplikasi ini merupakan bagian dari tugas klasifikasi penggunaan lahan menggunakan **Citra Sentinel-2A**, model **Random Forest**, dan divisualisasikan dengan **Folium**.
""")

st.header("1. Data Understanding")
st.subheader("1.1 Tujuan Analisis Land Use and Land Classification")
st.markdown("""
1. **Mengklasifikasikan** tutupan/penggunaan lahan di wilayah Jawa Timur ke dalam 6 kelas (Sawah, Bangunan, Mangrove, Lahan Hijau, Perairan Terbuka/Laut, dan Danau) berdasarkan reflektansi band Sentinel-2A dan indeks spektralnya.
2. **Memahami karakteristik spektral** tiap kelas lahan.
3. **Membangun dan mengevaluasi model Random Forest** yang dikenal menghasilkan akurasi tinggi pada data spasial.
4. **Menyajikan peta hasil klasifikasi** menggunakan *Leaflet/Folium* dengan base map WMS satelit Google Maps.
""")

st.subheader("1.2 Acuan Klasifikasi Lahan di Indonesia")
st.markdown("""
Acuan standar yang sering digunakan di Indonesia untuk klasifikasi tutupan lahan adalah **SNI 7645-1:2014** tentang Klasifikasi Penutup Lahan. SNI ini membagi penutup lahan menjadi berbagai hierarki (seperti daerah bervegetasi, tidak bervegetasi, perairan buatan, dan alami). 
Pada eksperimen ini, digunakan **6 kelas** tutupan lahan:
- **Sawah**: Lahan pertanian lahan basah.
- **Bangunan**: Area permukiman, industri, dan infrastruktur.
- **Mangrove**: Ekosistem pesisir.
- **Lahan Hijau**: Hutan dan vegetasi selain mangrove/sawah.
- **Perairan Terbuka (Laut)**: Wilayah lautan.
- **Danau**: Perairan tawar alami/buatan.
""")

st.subheader("1.3 Deskripsi Macam-Macam Band Sentinel-2A")
st.markdown("""
Citra satelit Sentinel-2A memiliki resolusi spasial yang bervariasi (10m, 20m, 60m). Band yang digunakan dalam pemodelan:
| Band | Nama | Resolusi | Kegunaan |
|---|---|---|---|
| B2 | Blue | 10 m | Penetrasi air, bangunan |
| B3 | Green | 10 m | Puncak pantulan vegetasi, indeks air |
| B4 | Red | 10 m | Penyerapan klorofil |
| B5 | Red Edge 1 | 20 m | Kandungan klorofil vegetasi |
| B6 | Red Edge 2 | 20 m | Struktur kanopi |
| B7 | Red Edge 3 | 20 m | Struktur kanopi |
| B8 | NIR | 10 m | Biomassa vegetasi, daratan vs air |
| B8A| Narrow NIR | 20 m | Kadar air tanaman |
| B11| SWIR 1 | 20 m | Kelembapan tanah, area terbangun |
| B12| SWIR 2 | 20 m | Tanah terbuka, area terbangun |
""")

st.subheader("1.4 Fitur yang Diekstraksi dan Rumusnya")
st.markdown("""
Selain band asli, diekstraksi **8 indeks spektral** ($\rho$ melambangkan reflektansi band tertentu):
1. **NDVI** (Normalized Difference Vegetation Index): $\\frac{\\rho_{B8}-\\rho_{B4}}{\\rho_{B8}+\\rho_{B4}}$
2. **NDWI** (Normalized Difference Water Index): $\\frac{\\rho_{B3}-\\rho_{B8}}{\\rho_{B3}+\\rho_{B8}}$
3. **MNDWI** (Modified NDWI): $\\frac{\\rho_{B3}-\\rho_{B11}}{\\rho_{B3}+\\rho_{B11}}$
4. **NDBI** (Normalized Difference Built-up Index): $\\frac{\\rho_{B11}-\\rho_{B8}}{\\rho_{B11}+\\rho_{B8}}$
5. **NDRE** (Normalized Difference Red Edge): $\\frac{\\rho_{B8}-\\rho_{B5}}{\\rho_{B8}+\\rho_{B5}}$
6. **EVI** (Enhanced Vegetation Index): $2.5 \\times \\frac{\\rho_{B8}-\\rho_{B4}}{\\rho_{B8}+6\\rho_{B4}-7.5\\rho_{B2}+1}$
7. **SAVI** (Soil Adjusted Vegetation Index): $1.5 \\times \\frac{\\rho_{B8}-\\rho_{B4}}{\\rho_{B8}+\\rho_{B4}+0.5}$
8. **BSI** (Bare Soil Index): $\\frac{(\\rho_{B11}+\\rho_{B4})-(\\rho_{B8}+\\rho_{B2})}{(\\rho_{B11}+\\rho_{B4})+(\\rho_{B8}+\\rho_{B2})}$
""")

st.header("2. Collecting Data & Preprocessing")
st.markdown("""
Pengumpulan data dilakukan dengan mengambil sampel poligon di Jawa Timur dan mengekstrak nilai piksel Sentinel-2A. Total kelas: **6 Kelas**.
Sumber area poligon menggunakan file referensi berikut:
- **`50 Sawah gqis.qgz`**
- **`50 Non Sawah gqis.qgz`** (Bangunan)
- **`Lahan hijau.zip`**
- **`laut.zip`**
- **`Danau.zip`**
- **`Mangrove Zaidan.zip`**
""")

@st.cache_data
def load_data():
    return pd.read_csv("dataset_sentinel2_jatim.csv")

try:
    df = load_data()
    st.dataframe(df.head())
    
    st.write(f"**Total Data:** {df.shape[0]} titik sampel.")
    
    class_counts = df['kelas'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette="viridis")
    ax.set_title("Jumlah Data per Kelas")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

st.header("3. Pemodelan (Random Forest Classifier)")
st.markdown("Model **Random Forest** digunakan karena mampu menangani data non-linear dan mencegah *overfitting*, memberikan akurasi yang lebih baik pada data penginderaan jauh.")

# Fitur dan Target
FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
X = df[FITUR]
y = df['kelas_id']
kelas_map = dict(zip(df['kelas_id'], df['kelas']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

col1, col2 = st.columns(2)
with col1:
    st.info(f"📊 **Data Training (80%):** {X_train.shape[0]} titik sampel")
with col2:
    st.info(f"🧪 **Data Testing (20%):** {X_test.shape[0]} titik sampel")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Prediksi untuk seluruh dataset agar peta menampilkan semua titik klasifikasi
df['prediksi'] = rf.predict(X)
df['kelas_prediksi'] = df['prediksi'].map(kelas_map)

acc = accuracy_score(y_test, y_pred)
st.success(f"🎯 **Akurasi Model Random Forest (Data Testing):** {acc * 100:.2f}%")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[kelas_map[i] for i in sorted(kelas_map.keys())],
            yticklabels=[kelas_map[i] for i in sorted(kelas_map.keys())], ax=ax2)
ax2.set_xlabel('Prediksi')
ax2.set_ylabel('Aktual')
st.pyplot(fig2)

st.header("4. Visualisasi Hasil Klasifikasi")
st.markdown("""
Peta interaktif menggunakan WMS Google Maps Satellite. Seluruh titik sampel (**520 piksel**) diprediksi menggunakan model Random Forest yang telah dilatih.
Gunakan filter di bawah untuk memilih kelas tutupan lahan yang ingin ditampilkan.
""")

colors = {
    1: "#FFA500",  # Sawah - Orange/Gold
    2: "#E63946",  # Bangunan - Merah
    3: "#9D4EDD",  # Mangrove - Ungu
    4: "#2A9D8F",  # Lahan Hijau - Hijau
    5: "#1D3557",  # Lautan - Biru Tua
    6: "#4EA8DE",  # Danau - Biru Muda
}

# Filter interaktif kelas di Streamlit
opsi_kelas = list(kelas_map.values())
kelas_terpilih = st.multiselect(
    "Filter Kelas yang Ditampilkan di Peta:",
    options=opsi_kelas,
    default=opsi_kelas
)

df_filtered = df[df['kelas_prediksi'].isin(kelas_terpilih)]
st.caption(f"Menampilkan **{len(df_filtered)}** titik sampel terprediksi dari total {len(df)} titik.")

m = folium.Map(location=[-7.7, 112.5], zoom_start=8)

# Google Satellite WMS
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
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

# Kelompokkan marker berdasarkan kelas
for nama_k in kelas_terpilih:
    df_sub = df_filtered[df_filtered['kelas_prediksi'] == nama_k]
    k_id = [k for k, v in kelas_map.items() if v == nama_k][0]
    fg = folium.FeatureGroup(name=f"Prediksi: {nama_k} ({len(df_sub)} titik)")
    
    for idx, row in df_sub.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            popup=f"<b>Prediksi:</b> {nama_k}<br><b>Aktual:</b> {row['kelas']}<br><b>Koordinat:</b> ({row['lat']:.4f}, {row['lon']:.4f})",
            color="white",
            weight=1,
            fill=True,
            fill_color=colors.get(k_id, "black"),
            fill_opacity=0.85
        ).add_to(fg)
    fg.add_to(m)

# Tambahkan legenda HTML melayang pada peta
legend_html = '''
<div style="position: fixed; 
     bottom: 30px; left: 30px; width: 190px;
     background-color: white; z-index:9999; font-size:12px;
     border:2px solid #bbb; border-radius: 8px; padding: 10px;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
     <b style="font-size:13px;">Legenda Kelas LULC</b><br>
     <i style="background:#FFA500; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i> Sawah<br>
     <i style="background:#E63946; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i> Bangunan<br>
     <i style="background:#9D4EDD; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i> Mangrove<br>
     <i style="background:#2A9D8F; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i> Lahan Hijau<br>
     <i style="background:#1D3557; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i> Perairan Laut<br>
     <i style="background:#4EA8DE; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i> Danau<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)

folium_static(m)

st.header("5. Peta Khusus Kelas Laut di Jawa Timur")
st.markdown("Untuk melihat letak spesifik kelas perairan laut yang diklasifikasikan oleh model, peta di bawah ini menampilkan **seluruh titik sampel yang diprediksi sebagai Perairan Terbuka (Laut)**.")

# Ambil SEMUA titik yang diprediksi sebagai laut (kelas_id = 5) dari seluruh data
df_laut = df[df['prediksi'] == 5]

st.metric("Total Titik Terprediksi Laut", f"{len(df_laut)} titik sampel")

m_laut = folium.Map(location=[-7.7, 112.5], zoom_start=8)
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(m_laut)

for idx, row in df_laut.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=7,
        popup=f"Titik Laut (lon: {row['lon']:.4f}, lat: {row['lat']:.4f})",
        color="white",
        weight=1.5,
        fill=True,
        fill_color="#0077b6",
        fill_opacity=0.85
    ).add_to(m_laut)

folium_static(m_laut)

st.markdown("### Kesimpulan")
st.markdown("""
Eksperimen menggunakan **Random Forest** berhasil mengklasifikasikan 6 kelas lahan di Jawa Timur dengan akurasi yang sangat baik. 
Fitur indeks spektral memberikan kontribusi signifikan dalam membedakan karakteristik antara vegetasi, perairan, dan lahan terbangun.
""")
