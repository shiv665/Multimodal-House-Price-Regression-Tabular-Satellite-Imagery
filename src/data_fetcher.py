import os
import argparse
import time
import math
import requests
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from src.config import cfg

def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates for the given zoom level."""
    # a complex formula to convert lat/lon to tile x,y
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def esri_url(lat, lon, zoom):
    """
    Generate ESRI World Imagery tile URL (FREE - no API key needed).
    Uses ArcGIS World Imagery service.
    """
    x, y = lat_lon_to_tile(lat, lon, zoom)
    return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

def fetch_image(lat, lon, max_retries=5, retry_delay=2):
    """
    Fetch satellite image from ESRI World Imagery (FREE).
    Downloads a tile and returns it as PNG bytes.
    Includes retry logic for network errors.
    Uses OpenCV for image processing.
    """
    url = esri_url(lat, lon, cfg.zoom)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30, headers=headers)
            if r.status_code == 200:
                # Decode image bytes to numpy array using OpenCV
                img_array = np.frombuffer(r.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"Failed to decode image")
                    return None
                
                # Resize to desired tile size using OpenCV
                img = cv2.resize(img, (cfg.tile_size, cfg.tile_size), interpolation=cv2.INTER_LANCZOS4)
                
                # Encode to PNG bytes using OpenCV
                success, buffer = cv2.imencode('.png', img)
                if success:
                    return buffer.tobytes()
                return None
            elif r.status_code == 429:  # Rate limited
                wait_time = retry_delay * (attempt + 1)
                time.sleep(wait_time)
                continue
            else:
                print(f"Failed to fetch image: HTTP {r.status_code}")
                return None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Error fetching image after {max_retries} attempts: {e}")
    return None

def download(df, lat_col="lat", lon_col="long", id_col="id"):
    """Download satellite images. Skips already downloaded images (cached locally)."""
    os.makedirs(cfg.image_dir, exist_ok=True)
    paths = {}
    skipped = 0
    downloaded = 0
    failed_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading satellite tiles"):
        lat, lon, pid = row[lat_col], row[lon_col], row[id_col]
        fname = f"{pid}_{lat:.5f}_{lon:.5f}.png"
        fpath = os.path.join(cfg.image_dir, fname)
        
        # Check cache - skip if already downloaded
        if os.path.exists(fpath):
            paths[pid] = fpath
            skipped += 1
            continue
        
        # Fetch from API
        content = fetch_image(lat, lon)
        if content is None:
            failed_list.append((lat, lon, pid, fpath))
            continue
        
        # Save to disk (cache)
        with open(fpath, "wb") as f:
            f.write(content)
        paths[pid] = fpath
        downloaded += 1
        time.sleep(0.1)  # polite pause
    
    # Retry failed downloads
    if failed_list:
        print(f"\nRetrying {len(failed_list)} failed downloads...")
        time.sleep(2)  # Wait before retrying
        retry_success = 0
        
        for lat, lon, pid, fpath in tqdm(failed_list, desc="Retrying failed"):
            time.sleep(0.5)  # Slower pace for retries
            content = fetch_image(lat, lon, max_retries=3, retry_delay=3)
            if content:
                with open(fpath, "wb") as f:
                    f.write(content)
                paths[pid] = fpath
                retry_success += 1
        
        final_failed = len(failed_list) - retry_success
        print(f"Retry results: {retry_success} recovered, {final_failed} still failed")
    else:
        final_failed = 0
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  - Already cached (skipped): {skipped}")
    print(f"  - Newly downloaded: {downloaded}")
    print(f"  - Failed: {final_failed}")
    print(f"  - Total images: {len(paths)}")
    print(f"  - Saved to: {cfg.image_dir}/")
    print(f"{'='*50}")
    
    return paths

def main():
    train_df = pd.read_excel(cfg.train_xlsx)
    test_df = pd.read_excel(cfg.test_xlsx)
    download(pd.concat([train_df, test_df], axis=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()