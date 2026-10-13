import math
import io
import os
import requests
import numpy as np
from PIL import Image

def get_xy(lon, lat, z):
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y

def lon_from_x(x, z):
    n = 2**z
    return x / n * 360.0 - 180.0

def lat_from_y(y, z):
    n = 2**z
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))

z = 9
lon_min, lon_max = 110.85, 114.65
lat_min, lat_max = -8.85, -6.65

x0, y0 = get_xy(lon_min, lat_max, z)
x1, y1 = get_xy(lon_max, lat_min, z)
xa, xb = int(x0), int(x1)
ya, yb = int(y0), int(y1)

tile_w = (xb - xa + 1) * 256
tile_h = (yb - ya + 1) * 256
dem_mosaic = np.zeros((tile_h, tile_w), dtype=np.float32)

print(f"Downloading { (xb-xa+1)*(yb-ya+1) } DEM tiles (z={z})...")
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount('https://', adapter)

headers = {"User-Agent": "Mozilla/5.0"}
for y in range(ya, yb + 1):
    for x in range(xa, xb + 1):
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        try:
            r = session.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                arr = np.array(img, dtype=np.float32)
                # Terrarium: elevation = (R * 256 + G + B / 256) - 32768
                elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
                ox = (x - xa) * 256
                oy = (y - ya) * 256
                dem_mosaic[oy:oy+256, ox:ox+256] = elev
            else:
                print(f"Failed {x},{y}: code {r.status_code}")
        except Exception as e:
            print(f"Error {x},{y}: {e}")

print("DEM Mosaic shape:", dem_mosaic.shape)
print("Min elev:", np.min(dem_mosaic), "Max elev:", np.max(dem_mosaic))
np.save("scratch/dem_mosaic_z9.npy", dem_mosaic)
print("Saved DEM mosaic to scratch/dem_mosaic_z9.npy")
