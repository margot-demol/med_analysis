import numpy as np
import pandas as pd
import xarray as xr
import os
from glob import glob

#from pyproj import Geod, Proj
import geopandas as gpd
import shapely as shp
from shapely.geometry import Polygon
import pyproj
from pyproj import Geod

from cstes import swot_dir, swot_dir_2km, drifters_dir, get_proj, lonlat2xy, zarr_dir

def browse_swot_250():
    """ browse SWOT files """
    passes = [3, 16]
    
    D = []
    for p in passes:
        files = sorted(glob(os.path.join(swot_dir, f"{p}_*.zarr")))
        for f in files:
            c = int(f.split("/")[-1].replace(".zarr","").split("_")[1])
            if (p==3)& (c in [568]) : continue #empty cycle_number
            if (p==16)& (c in [508,513, 534, 554, 568]) : continue #empty cycle_number
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

def browse_swot_2km():
    """ browse SWOT files """
    passes = [3, 16]
    
    D = []
    for p in passes:
        files = sorted(glob(os.path.join(swot_dir_2km, f"{p}_*.zarr")))
        for f in files:
            c = int(f.split("/")[-1].replace(".zarr","").split("_")[1])
            if (p==3)& (c in [568]) : continue #empty cycle_number
            if (p==16)& (c in [508,513, 534, 554, 568]) : continue #empty cycle_number
            t = xr.open_zarr(f).isel(num_lines=0)["time"].data.compute()[()]
            #print(f)
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

    e = da.isel(num_lines=rg(ny,diy), num_pixels=-1)
    X += list(e[x].data)
    Y += list(e[y].data)
    
    e = da.isel(num_lines=-1, num_pixels=rg(nx, dix))
    X += list(e[x].data[::-1])
    Y += list(e[y].data[::-1])
    
    e = da.isel(num_lines=rg(ny,diy), num_pixels=0)
    X += list(e[x].data[::-1])
    Y += list(e[y].data[::-1])
    
    #mpt = shp.MultiPoint([[x, y] for x, y in zip(X, Y)])
    #poly = mpt.convex_hull
    poly = shp.Polygon([[x, y] for x, y in zip(X, Y)])

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



# get grid orientation and metrics
def add_grid_metrics(ds):
    """ add grid spatial metrics """

    geod = Geod(ellps="WGS84")

    lon, lat = ds.longitude, ds.latitude
    dims = lon.dims
    
    # d/dx where x is cross-track
    az12, az21, dx = geod.inv(
        lon, lat, lon.shift(num_pixels=-1), lat.shift(num_pixels=-1),
    )
    
    ds = ds.assign_coords(dx=(dims, dx), phi=(dims, az12*np.pi/180))
    
    ds["dx"] = (
        ds["dx"]
        .ffill("num_pixels")
        .where(ds["duacs_editing_flag"]<5)
    )

    # phi_lon is cross-track direction from north
    ds["phix"] = (
        ds["phi"]
        .ffill("num_pixels")
        .where(ds["duacs_editing_flag"]<5)
    )

    # d/dy where y is along-track
    az12, az21, dy = geod.inv(
        lon, lat, lon.shift(num_lines=-1), lat.shift(num_lines=-1),
    )
    
    # phiy is along-track direction from north
    ds = ds.assign_coords(phi=(dims, az12*np.pi/180))
    ds["phiy"] = (
        ds["phi"]
        .ffill("num_pixels")
        .where(ds["duacs_editing_flag"]<5)
    )
    
    ds = ds.assign_coords(dy=(dims, dy))
    
    ds["dy"] = (
        ds["dy"]
        .ffill("num_lines")
        .where(ds["duacs_editing_flag"]<5)
    )
    ds['phi'] = ds['phi'].where(ds["duacs_editing_flag"]<5)
    return ds


"""
______________
COMPUTE GRADIENT 
______________
"""
# Compute ggd with "naive" approach, noise will have an impact
g = 9.80665

ggd_variables = ['cvl_mean_dynamic_topography_cnes_cls_22',
                 'cvl_mean_sea_surface_cnes_22_hybrid',
                 'cvl_ocean_tide_fes_2022',
                 'cvl_ssha_reference',
                 'duacs_ssha_karin_2_calibrated',
                 'duacs_ssha_karin_2_filtered',]

def gradient_naive(dss, ggd_variables=None):
    g=9.81
    if ggd_variables : dss_ggd = dss[ggd_variables]
    
    dx = float(dss_ggd.dx.mean())
    dy = float(dss_ggd.dy.mean())

    dss_ggdx = xr.Dataset()
    dss_ggdy = xr.Dataset()
    
    dx = dss_ggd.dx.mean() # meters
    dy = dss_ggd.dy.mean()
    
    dss_ggdx = (g*dss_ggd.differentiate("num_pixels")/dx).rename({v:'ggdx_'+v for v in ggd_variables})
    dss_ggdy = (g*dss_ggd.differentiate("num_lines")/dy).rename({v:'ggdy_'+v for v in ggd_variables})

    dss_ggdxx = (dss_ggdx.differentiate("num_pixels")/dx).rename({'ggdx_'+v:'ggdxx_'+v for v in ggd_variables})
    dss_ggdyy = (dss_ggdy.differentiate("num_lines")/dy).rename({'ggdy_'+v:'ggdyy_'+v for v in ggd_variables})
    dss_ggdxy = (dss_ggdx.differentiate("num_lines")/dy).rename({'ggdx_'+v:'ggdxy_'+v for v in ggd_variables})
    
    return xr.merge([dss_ggdx, dss_ggdy, dss_ggdxx, dss_ggdyy, dss_ggdxy]).where(~dss['duacs_ssha_karin_2_filtered'].isnull())

def strain_vorticity(dsg) : 
    dsg['f'] =  2 * 2 * np.pi / 86164.1 * np.sin(dsg.latitude * np.pi / 180)
    dsg['vorticity'] = (dsg['ggdxx_duacs_ssha_karin_2_filtered'] 
                        + dsg['ggdxx_cvl_mean_dynamic_topography_cnes_cls_22']
                        + dsg['ggdxx_cvl_ocean_tide_fes_2022']
                        + dsg['ggdyy_duacs_ssha_karin_2_filtered']
                        + dsg['ggdyy_cvl_mean_dynamic_topography_cnes_cls_22']
                        + dsg['ggdyy_cvl_ocean_tide_fes_2022'])/dsg['f']
    
    dsg['sigma_s'] = (dsg['ggdxx_duacs_ssha_karin_2_filtered'] 
                        + dsg['ggdxx_cvl_mean_dynamic_topography_cnes_cls_22']
                        + dsg['ggdxx_cvl_mean_dynamic_topography_cnes_cls_22']
                        - dsg['ggdyy_duacs_ssha_karin_2_filtered']
                        - dsg['ggdyy_cvl_mean_dynamic_topography_cnes_cls_22']
                        - dsg['ggdyy_cvl_ocean_tide_fes_2022'])/dsg['f']
    
    dsg['sigma_n'] = -2*(dsg['ggdxy_duacs_ssha_karin_2_filtered'] 
                        + dsg['ggdxy_cvl_mean_dynamic_topography_cnes_cls_22']
                        + dsg['ggdxy_cvl_ocean_tide_fes_2022'])/dsg['f']


# gaussian derivative
from scipy.ndimage import gaussian_filter

def gradient_gauss(dss, cutoff, ggd_variables=None, **kwargs):
    
    if ggd_variables : dss_ggd = dss[ggd_variables]
    else : dss_ggd = dss
    
    dx = float(dss_ggd.dx.mean())
    dy = float(dss_ggd.dy.mean())

    dss_ggdx = xr.Dataset()
    dss_ggdy = xr.Dataset()
    
    for v in dss_ggd : 
        da = dss_ggd[v].interpolate_na(dim='num_pixels').interpolate_na(dim='num_lines')# prevent nan to spread
        
        # cross-track
        i = da.get_axis_num("num_pixels")
        order = [0,0]
        order[i] = 1
        dss_ggdx['ggdx_'+v] = g*(xr.DataArray(gaussian_filter(da, sigma=cutoff/dx, order=order, **kwargs), dims=da.dims)/dx).where(~dss['duacs_ssha_karin_2_filtered'].isnull())
        #da_dx = da_dx.where(da)
        
        # along-track
        i = da.get_axis_num("num_lines")
        order = [0,0]
        order[i] = 1
        dss_ggdy['ggdy_'+v] = g*(xr.DataArray(gaussian_filter(da, sigma=cutoff/dy, order=order, **kwargs), dims=da.dims)/dy).where(~dss['duacs_ssha_karin_2_filtered'].isnull())
        
    return xr.merge([dss_ggdx, dss_ggdy])

def secondderivative_gauss(dss, cutoff, ggd_variables=None, **kwargs):
    g=9.81
    # gaussian derivative
    from scipy.ndimage import gaussian_filter

    if ggd_variables : dss_ggd = dss[ggd_variables]
    else : dss_ggd = dss
    
    dx = float(dss_ggd.dx.mean())
    dy = float(dss_ggd.dy.mean())

    dss_ggdx = xr.Dataset()
    dss_ggdy = xr.Dataset()
    dss_ggdxx = xr.Dataset()
    dss_ggdyy = xr.Dataset()
    dss_ggdxy = xr.Dataset()
    
    
    for v in dss_ggd : 
        da = dss_ggd[v].interpolate_na(dim='num_pixels').interpolate_na(dim='num_lines')# prevent nan to spread
        
        # cross-track
        i = da.get_axis_num("num_pixels")
        order = [0,0]
        order[i] = 1
        dss_ggdx['ggdx_'+v] = g*(xr.DataArray(gaussian_filter(da, sigma=cutoff/dx, order=order, **kwargs), dims=da.dims)/dx)
        dss_ggdx['ggdxx_'+v] = (xr.DataArray(gaussian_filter(dss_ggdx['ggdx_'+v], sigma=cutoff/dx, order=order, **kwargs), dims=da.dims)/dx)

        
        # along-track
        i = da.get_axis_num("num_lines")
        order = [0,0]
        order[i] = 1
        dss_ggdy['ggdy_'+v] = g*(xr.DataArray(gaussian_filter(da, sigma=cutoff/dy, order=order, **kwargs), dims=da.dims)/dy)
        dss_ggdyy['ggdyy_'+v] = (xr.DataArray(gaussian_filter(dss_ggdy['ggdy_'+v], sigma=cutoff/dy, order=order, **kwargs), dims=da.dims)/dy)
        dss_ggdxy['ggdxy_'+v] = (xr.DataArray(gaussian_filter(dss_ggdx['ggdx_'+v], sigma=cutoff/dy, order=order, **kwargs), dims=da.dims)/dy)
              
    return xr.merge([dss_ggdx, dss_ggdy, dss_ggdxx, dss_ggdyy, dss_ggdxy]).where(~dss['duacs_ssha_karin_2_filtered'].isnull())


# Variance estimation
def etaf(dfs):
    dfs['ggdx_etaf'] = dfs.ggdx_duacs_ssha_karin_2_filtered +dfs.ggdx_cvl_mean_dynamic_topography_cnes_cls_22+dfs.ggdx_cvl_ocean_tide_fes_2022
    dfs['ggdy_etaf'] = dfs.ggdy_duacs_ssha_karin_2_filtered + dfs.ggdy_cvl_mean_dynamic_topography_cnes_cls_22+dfs.ggdy_cvl_ocean_tide_fes_2022



"""
______________
INTERP 
______________
"""
import pyinterp
mesh = pyinterp.RTree()

def interp_one_dataarray(da, new_lon, new_lat):
    lons = da.longitude.compute().data.flatten()
    lats = da.latitude.compute().data.flatten()
    data = da.compute().data.flatten()

    mesh.packing(np.vstack((lons, lats)).T, data)
    mx = new_lon
    my = new_lat

    idw, neighbors = mesh.inverse_distance_weighting(
        np.vstack((mx.ravel(), my.ravel())).T,
        within=False,  # Extrapolation is forbidden
        k=5,  # We are looking for at most 11 neighbors
        num_threads=0,
    )
    idw = idw.reshape(mx.shape)
    return idw

def interp_dss(dss, new_lon, new_lat):
    df_interp = pd.DataFrame()
    df_interp['longitude'] = new_lon
    df_interp['latitude'] = new_lat
    for v in dss : 
        da = dss[v]
        df_interp[v]=interp_one_dataarray(da, new_lon, new_lat)
    return df_interp

"""
______________
COLOC 
______________
"""

def coloc_swot_cycle_swath(dfr, dfs, cycle, swath, cutoff, ggd_variables, variables, method_gradient='gauss'):
    # select drifter point
    dfr_ = dfr.where((dfr.pass_number==swath)&(dfr.cycle_number==cycle)).dropna()
    dss = xr.open_dataset(dfs.where((dfs.pass_number==swath)&(dfs.cycle_number==cycle)).dropna().file.values[0])
    dss = add_grid_metrics(dss).reset_coords(['phi'])
    dss = dss.where(dss.duacs_editing_flag==0) # ATTENTION SELECT pseudo good quality
    if method_gradient =='naive' :
        dss_ggd= gradient_naive(dss, ggd_variables=ggd_variables)
        strain_vorticity(dss_ggd)
    if method_gradient == 'gauss' :
        dss_ggd = gradient_gauss(dss, cutoff, ggd_variables)
    dss = xr.merge([dss[variables], dss_ggd])
    df_interp = interp_dss(dss, dfr_.longitude.values, dfr_.latitude.values)
    df_out = pd.concat([dfr_.reset_index()[['row_number', 'longitude']].set_index('longitude'), df_interp.set_index('longitude')], axis=1).reset_index().set_index('row_number')
    rotate_ggd(df_out, ggd_variables)
    return df_out

def coloc_swot(dfr, dfs, cutoff, ggd_variables, variables, method_gradient='gauss'):
    DF = []
    for swath in dfr.pass_number.unique() : 
        for cycle in dfr.where(dfr.pass_number==swath).dropna().cycle_number.unique():
            DF.append(coloc_swot_cycle_swath(dfr, dfs, cycle,  swath, cutoff,  ggd_variables, variables, method_gradient))
            print(cycle)
    return pd.concat(DF)
    
"""
______________
ROTATE 
______________
"""
def rotate(x, y, phi):
    return np.cos(phi)*x - np.sin(phi)*y, np.sin(phi)*x + np.cos(phi)*y
def rotate_ggd(df, ggd_variables):
    for v in ggd_variables : 
        df['ggde_'+v], df['ggdn_'+v] = rotate(df['ggdx_'+v], df['ggdy_'+v], df['phi'])

