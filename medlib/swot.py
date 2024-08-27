import numpy as np
import pandas as pd
import xarray as xr
import os
from glob import glob

#from pyproj import Geod, Proj
import geopandas as gpd
import shapely as shp
from shapely.geometry import Polygon


from medlib.cstes import swot_dir
def browse_swot_250():
    """ browse SWOT files """
    passes = [3, 16]
    
    D = []
    for p in passes:
        files = sorted(glob(os.path.join(swot_dir, f'L3_250', f"{p}_*.zarr")))
        for f in files:
            c = int(f.split("/")[-1].replace(".zarr","").split("_")[1])
            t = xr.open_zarr(f).isel(num_lines=0)["time"].data.compute()[()]
            D.append(dict(cycle_number=c, pass_number=p, file=f, time=t))

    df = pd.DataFrame(D)
    df["day"] = df["time"].dt.floor("1d")
    
    df['dt_before'] = df.groupby('pass_number').time.diff()/2
    mask = df.dt_before.isnull()
    df.loc[mask, 'dt_before'] = df.dt_before.median()

    df['dt_after'] = -df.groupby('pass_number').time.diff(periods = -1)/2
    mask = df.dt_after.isnull()
    df.loc[mask, 'dt_after'] = df.dt_after.median()

    df['start_time_cut'] = df.time-df.dt_before

    df['end_time_cut'] = df.time+df.dt_after
    df.drop(columns=['dt_before', 'dt_after'], inplace=True)
    
    return df.set_index(["day", "pass_number"])



def build_swath_polygon(da, x="x", y="y", dix = 50, diy = 50, side=None):
    """ build a shapely polygon of swot swath

    Parameters
    ----------
    da: xr.Dataset, xr.DataArray
        Input dataset where coordinates x/y are found
    x, y: str, optional
        Coordinates to use to define bounds
    dix, diy: int, optional
        Subsampling steps in num_pixels and num_lines directions
    side: str, optional
        take either the polygon for the full swath, or the right/left ones
    """

    # right swath
    if side=="left":
        da = da.isel(num_pixels=slice(0, 237))
    elif side=="right":
        da = da.isel(num_pixels=slice(282, -1))
            
    nx = da.num_pixels.size
    ny = da.num_lines.size
        
    rg = lambda size, di: list(range(0, size, di))+[size-1]
    
    X, Y = [], []
    
    e = da.isel(num_lines=0, num_pixels=rg(nx, dix))
    X += list(e[x].data)
    Y += list(e[y].data)
    
    e = da.isel(num_lines=-1, num_pixels=rg(nx, dix))
    X += list(e[x].data)
    Y += list(e[y].data)
    
    e = da.isel(num_lines=rg(ny,diy), num_pixels=0)
    X += list(e[x].data)
    Y += list(e[y].data)
    
    e = da.isel(num_lines=rg(ny,diy), num_pixels=-1)
    X += list(e[x].data)
    Y += list(e[y].data)
    
    mpt = shp.MultiPoint([[x, y] for x, y in zip(X, Y)])
    poly = mpt.convex_hull

    return poly

def add_mask_inside_swot(ds_swot, ds_drifters):
    # build swath polygons
    #both = build_swath_polygon(swot, x="longitude", y="latitude", dix = 50, diy = 50, side=None)
    left = build_swath_polygon(ds_swot, x="longitude", y="latitude", dix = 50, diy = 50, side="left")
    right = build_swath_polygon(ds_swot, x="longitude", y="latitude", dix = 50, diy = 50, side="right")
    # stack dimensions
    drs = ds_drifters.stack(points=("datetime", "drifter_id"))
    lon, lat = drs.longitude.fillna(0.).values, drs.latitude.fillna(0.).values
    # test if points inside right swath
    drs["inside_right"] = (
        "points", 
        np.array([right.contains(shp.Point(_lon, _lat)) for _lon, _lat in zip(lon, lat)]),
    )
    # test if points inside left swath
    drs["inside_left"] = (
        "points", 
        np.array([left.contains(shp.Point(_lon, _lat)) for _lon, _lat in zip(lon, lat)]),
    )
    # put mask inside original dataset
    ds_drifters["inside_left"] = drs["inside_left"].unstack()
    ds_drifters["inside_right"] = drs["inside_right"].unstack()
    return ds_drifters