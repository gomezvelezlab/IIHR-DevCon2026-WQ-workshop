import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import pandas as pd
import xarray as xr
import rasterio
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import zipfile
import datetime
import shutil


def list_zip_files(target_url):
    try:
        # Fetch the page content
        response = requests.get(target_url)
        response.raise_for_status()  # Check for errors
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all <a> tags and filter for .zip extensions
        zip_files_names = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.lower().endswith('.zip'):
                zip_files_names.append(href)
        
        # Return the results
        return zip_files_names
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")

def download_usgs_data(file_names, base_url, target_dir, force_download = False):
    
    # Create the download directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Starting download of {len(file_names)} files...")

    # Download each file with a progress bar
    for file_name in file_names:
        file_url = base_url + file_name
        file_path = os.path.join(target_dir, file_name)

        # Check if file already exists to skip re-downloading
        if os.path.exists(file_path) and (not force_download):
            print(f"Skipping {file_name} (already exists).")
            continue

        # Stream the download
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

def read_tiff_data(filename, data_dir):

    data = rioxarray.open_rasterio(os.path.join(data_dir, filename))
    nodata = data.rio.nodata

    # Mask nodata values before dividing
    data_masked = data.where(data != nodata)
    data_scaled = data_masked / 1000

    # Fill back nodata where it was masked
    data_scaled = data_scaled.where(~data_masked.isnull(), other=nodata)
    data_scaled.rio.write_nodata(nodata, inplace=True)

    data = data_scaled

    # Load data into memory to close the file handle
    data = data.load()

    return data

def get_bbox(bbox_coords = None):

    if bbox_coords is None:
        # Create a bounding box polygon
        lat_min, lat_max, lon_min, lon_max = 40.17, 45.83, -98.04, -89.95
    else:
        lat_min, lat_max, lon_min, lon_max = bbox_coords

    # Create a bounding box polygon
    bbox = gpd.GeoDataFrame({
        'geometry': [box(lon_min, lat_min, lon_max, lat_max)]
    }, crs="EPSG:4326")

    return bbox

def clip_data(filename, data_dir, bbox_coords = None): 

    bbox = get_bbox(bbox_coords)

    data = read_tiff_data(filename, data_dir)
    # Clip data to the bounding box
    data_clipped = data.rio.clip(bbox.geometry, bbox.crs, drop=True)

    return data_clipped

def extract_zip_file(filename, extract_dir, verbose = False):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        if verbose:
            print(f"---- Extracted {filename} to {extract_dir}")

def process_actual_et_zip_files(filenames_level_0, data_dir):
    
    filenames_level_0.sort()

    for filename_level_0 in filenames_level_0:

        full_filename_level_0 = os.path.join(data_dir, filename_level_0)
        full_extract_dir_level_0 = os.path.join(data_dir, full_filename_level_0.replace('.zip', ''))

        if not os.path.exists(full_extract_dir_level_0):
            os.makedirs(full_extract_dir_level_0, exist_ok=True)
            extract_zip_file(full_filename_level_0, full_extract_dir_level_0)
        
        filenames_level_1 = os.listdir(full_extract_dir_level_0)
        filenames_level_1.sort()

        data_list = []
        yeardoy_list = []

        for idx, filename_level_1 in enumerate(filenames_level_1):

            full_filename_level_1 = os.path.join(full_extract_dir_level_0, filename_level_1)
            full_filename_level_1 = [ff for ff in full_filename_level_1 if ff.endswith(".zip")]
            full_extract_dir_level_1 = os.path.join(data_dir, full_filename_level_1.replace('.zip', ''))

            if not os.path.exists(full_extract_dir_level_1):
                os.makedirs(full_extract_dir_level_1, exist_ok=True)
                extract_zip_file(full_filename_level_1, full_extract_dir_level_1)

            filename = [f for f in os.listdir(full_extract_dir_level_1) if f.endswith('.tif')][0]
            yeardoy = filename.split('.')[0].split('det')[-1]

            data = clip_data(filename, full_extract_dir_level_1, bbox_coords = None)

            data_list.append(data)
            yeardoy_list.append(yeardoy)

            # if idx == 2:
            #     break

        # Concatenate the data, and save to netcdf
        date_list = []
        for yeardoy in yeardoy_list:
            year = int(yeardoy[0:4])
            doy = int(yeardoy[4:])
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
            date_list.append(date)

        # Stack the data along a new "yeardoy" coordinate
        # Remove "band" dimension if it is present and equals 1, to simplify stack

        processed_data = []
        for d in data_list:
            # Reduce to 2D if needed
            if "band" in d.dims and d.sizes["band"] == 1:
                dataarray = d.isel(band=0, drop=True) #.to_array().squeeze()
                processed_data.append(dataarray)
            else:
                dataarray = d #.to_array().squeeze()
                processed_data.append(dataarray)

        # Stack them along the new dimension
        # Ensure all processed_data DataArrays have x and y coordinates,
        # then concatenate along the new "date" dimension, preserving spatial coordinates.
        data_xr = xr.concat(
            processed_data, 
            dim=xr.DataArray(date_list, dims='date', name='date')
        )
        data_xr = data_xr.to_dataset(name="actual_et_mm_day")

        data_xr.to_netcdf(os.path.join(data_dir, full_filename_level_0.replace('.zip', '_clipped_iowa.nc')))

        # Remove the extracted directory
        shutil.rmtree(full_extract_dir_level_0)

        print(f'Done wirh file {filename_level_0}')

if __name__ == "__main__":

    url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads/"
    zip_files_names = list_zip_files(url)

    if zip_files_names:
        print(f"Found {len(zip_files_names)} ZIP files:")
        for file in zip_files_names:
            print(file)
    else:
        print("No ZIP files found at this URL.")


