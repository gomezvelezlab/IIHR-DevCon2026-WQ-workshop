# -----------------------------------------------------------------------------
# Author: Jesus D. Gomez-Velez
# Email: jesus-gomezvelez@uiowa.edu
# Date: 2025-06-10
# Description: This module was coded by Jesus D. Gomez-Velez for processing AORC datasets.
# -----------------------------------------------------------------------------

import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

def project_geometry(geometry: gpd.GeoDataFrame, crs: str = "epsg:26915") -> gpd.GeoDataFrame:
    """
    Project a geometry to a given CRS.
    """
    return geometry.to_crs(crs)

def process_xarray_dataset(ds: xr.Dataset, crs: str = "epsg:26915", chunk_time: int = 1000) -> xr.Dataset:
    """
    Process an AORC dataset.
    Parameters
    ----------
    ds: xr.Dataset
    crs: str
    chunk_time: int
        The number of time steps to process at a time.
        Default is 1000.
        If the time dimension is less than 10000, the dataset is reprojected in one chunk.
        If the time dimension is greater than 10000, the dataset is reprojected in chunks of chunk_time time steps.

    Returns
    -------
    xr.Dataset
    """

    # ds_proj = ds.rio.reproject(crs)
    # Check if time dimension is large
    if len(ds.time) > 10000:
        # Reproject in chunks
        chunks = []
        for i in range(0, len(ds.time), chunk_time):
            ds_chunk = ds.isel(time=slice(i, min(i + chunk_time, len(ds.time))))
            ds_chunk_proj = ds_chunk.rio.reproject(crs)
            chunks.append(ds_chunk_proj)
        
        ds_proj = xr.concat(chunks, dim='time')
    else:
        ds_proj = ds.rio.reproject(crs)
        
    ds_proj.rio.write_crs(crs, inplace=True)
    ds_proj.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    ds_proj.coords['x'].attrs['long_name'] = 'x'
    ds_proj.coords['y'].attrs['long_name'] = 'y'
    ds_proj.coords['x'].attrs['units'] = 'meters'
    ds_proj.coords['y'].attrs['units'] = 'meters'
    return ds_proj


def plot_comparison_latlon_xy(
    ds_latlon: xr.Dataset,
    ds_xy: xr.Dataset,
    var: str = "APCP_surface",
    time_idx: int = 0,
    gdf_boundary_latlon: gpd.GeoDataFrame = None,
    gdf_boundary_xy: gpd.GeoDataFrame = None,
) -> None:
    """
    Plot a comparison of the latitude and longitude and the x and y coordinates.
    Parameters
    ----------
    ds_latlon: xr.Dataset
    ds_xy: xr.Dataset
    var: str
    time_idx: int
    gdf_boundary_latlon: gpd.GeoDataFrame
    gdf_boundary_xy: gpd.GeoDataFrame

    Returns
    -------
    None
    """

    the_palette = 'Blues'

    vmin, vmax = ds_latlon[var].isel(time=time_idx).min().values, ds_latlon[var].isel(time=time_idx).max().values

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs = axs.flatten()

    ax = axs[0]
    ds_latlon[var].isel(time=time_idx).plot(ax=ax, vmin=vmin, vmax=vmax, cmap=the_palette)
    gdf_boundary_latlon.boundary.plot(ax=ax, color='black', linewidth=1)
    gdf_boundary_xy.boundary.plot(ax=ax, color='black', linewidth=1)

    ax = axs[1]
    ds_xy[var].isel(time=time_idx).plot(ax=ax, vmin=vmin, vmax=vmax, cmap=the_palette)
    gdf_boundary_xy.boundary.plot(ax=ax, color='black', linewidth=1)

    fig.tight_layout()

    return fig, axs

def get_mask_from_geometry(gdf_boundary: gpd.GeoDataFrame, ds: xr.Dataset, is_latlon: bool = True) -> xr.DataArray:
    """
    Get a mask from a geometry.
    Parameters
    ----------
    gdf_boundary: gpd.GeoDataFrame
    ds: xr.Dataset
    is_latlon: bool

    Returns
    -------
    xr.Dataset
    The mask dataset with the mask variable.
    """

    var_name_ref = list(ds.data_vars)[0]
    ds_ref = ds[var_name_ref].isel(time=0)


    if is_latlon:
        
        ShapeMask = rasterio.features.geometry_mask(gdf_boundary.geometry,
                                            out_shape=(len(ds_ref.latitude), len(ds_ref.longitude)),
                                            transform=ds_ref.rio.transform(),
                                            invert=True,
                                            all_touched=True)        
        ShapeMask = xr.DataArray(ShapeMask , dims=("latitude", "longitude"))
    else:
        
        ShapeMask = rasterio.features.geometry_mask(gdf_boundary.geometry,
                                    out_shape=(len(ds_ref.y), len(ds_ref.x)),
                                    transform=ds_ref.rio.transform(),
                                    invert=True,
                                    all_touched=True)
        ShapeMask = xr.DataArray(ShapeMask , dims=("y", "x"))
    
    ds_mask = xr.Dataset({'mask': ShapeMask})
    
    return ds_mask

def clip_xarray_dataset_by_mask(ds: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """
    Clip an xarray dataset by a mask.
    Parameters
    ----------
    ds: xr.Dataset
    mask: xr.DataArray

    Returns
    -------
    xr.Dataset
    The clipped dataset.
    """
    return ds.where(mask['mask'], drop=True)

def process_aorc_dataset(ds: xr.Dataset, crs: str = None, clip_geometry: gpd.GeoDataFrame = None) -> xr.Dataset:
    """
    Process an AORC dataset.
    Parameters
    ----------
    ds: xr.Dataset
    crs: str
    clip_geometry: gpd.GeoDataFrame

    Returns
    -------
    xr.Dataset
    The processed dataset.
    """

    ds['APCP_surface'].attrs['units'] = 'mm/hr'

    ds['TMP_2maboveground'].data = ds['TMP_2maboveground'].data - 273.15
    ds['TMP_2maboveground'].attrs['units'] = 'C'

    if crs is not None:
        ds_proj = process_xarray_dataset(ds, crs)
        is_latlon = False
        if clip_geometry is not None:
            clip_geometry_proj = project_geometry(clip_geometry, crs)
        else:
            clip_geometry_proj = None
    else:
        ds_proj = ds
        clip_geometry_proj = clip_geometry
        is_latlon = True

    ds_mask = get_mask_from_geometry(clip_geometry_proj, ds_proj, is_latlon=is_latlon)

    if clip_geometry is not None:   
        ds_clipped = clip_xarray_dataset_by_mask(ds_proj, ds_mask)
        return ds_clipped, clip_geometry_proj
    else:
        return ds_proj, clip_geometry_proj

def get_aggregated_time_series_aorc(ds: xr.Dataset) -> pd.DataFrame:
    """
    Get an aggregated time series of an AORC dataset.
    Parameters
    ----------
    ds: xr.Dataset

    Returns
    -------
    pd.DataFrame
    """

    vars = list(ds.data_vars)

    df = pd.DataFrame({'time': ds.time.values})

    for var in vars:

        da = ds[var]
        name_col = da.attrs['long_name'] + '_'+ da.attrs['units']
        name_col = name_col.replace(' ', '_')


        var_mean = da.where(~np.isnan(da)).mean(dim=['y', 'x']).values
        var_std = da.where(~np.isnan(da)).std(dim=['y', 'x']).values

        df['mean_'+name_col] = var_mean
        df['std_'+name_col] = var_std

    return df

def plot_all_aorc_timeseries(df_agg_aorc: pd.DataFrame) -> None:
    """
    Plot all mean variables (time series) in df_agg_aorc.
    Parameters
    ----------
    df_agg_aorc: pd.DataFrame

    Returns
    -------
    None
    """
    # Plot all mean variables (time series) in df_agg_aorc

    # Exclude 'time' and any 'std_' columns for mean time series plots
    mean_cols = [col for col in df_agg_aorc.columns if col.startswith('mean_')]
    n_vars = len(mean_cols)

    fig, axes = plt.subplots(n_vars, 1, figsize=(15, 5 * n_vars), sharex=True)

    colors = sns.color_palette("husl", n_vars)

    if n_vars == 1:
        axes = [axes]

    for i, col in enumerate(mean_cols):
        ax = axes[i]
        sns.lineplot(data=df_agg_aorc, x='time', y=col, ax=ax, color=colors[i])
        var_label = col.replace('mean_', '').replace('_', ' ')
        ax.set_title(f'{var_label} mean within the watershed')
        ax.set_xlabel('Time')
        ax.set_ylabel(var_label)

    fig.tight_layout()

    return fig, axes