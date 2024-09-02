import numpy as np
import pandas as pd
import xarray as xr
import os
from glob import glob

#from pyproj import Geod, Proj
import geopandas as gpd
import shapely as shp
from shapely.geometry import Polygon

from cstes import swot_dir, drifters_dir, get_proj, lonlat2xy, zarr_dir

def browse_swot_250():
    """ browse SWOT files """
    passes = [3, 16]
    
    D = []
    for p in passes:
        files = sorted(glob(os.path.join(swot_dir, f"{p}_*.zarr")))
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


dfs = browse_swot_250().reset_index()
def swot_interp_grad_one(swath, cycle, lond, latd, row_number, variables, ggrad_variables) :
    g=9.81
    from scipy.interpolate import interp2d
    from scipy.interpolate import LinearNDInterpolator
    total_variables = variables + ggrad_variables
    # SWOT
    dss = xr.open_dataset(dfs.where((dfs.pass_number==swath)&(dfs.cycle_number==cycle)).dropna().file.values[0])[['longitude', 'latitude', 'time',]+ total_variables]
    
    # Create coordinates for num_lines and num_pixels
    dss['num_lines']=np.arange(len(dss.num_lines))
    dss['num_pixels']=np.arange(len(dss.num_pixels))
    
    # Create the interesting adt + corrections added back
    #dss['eta'] = dss.cvl_mean_dynamic_topography_cnes_cls_22 + dss.cvl_ocean_tide_fes_2022 + dss.duacs_ssha_karin_2_filtered
    #dss.eta.attrs={'long_name':'ADT+ocean tide correction', 'unit':'m'}
    
    # Projection in local swath ref (y in the num_pixels direction and x in the num_lines
    xd, yd = lonlat2xy(dss.attrs['lonc'],dss.attrs['latc'], dss.attrs['phi'], lond, latd, lon1=None, lat1=None)
    x, y = lonlat2xy(dss.attrs['lonc'],dss.attrs['latc'],  dss.attrs['phi'], dss.longitude, dss.latitude, lon1=None, lat1=None)
    dss['x']= xr.DataArray(x, dims=['num_lines', 'num_pixels'])
    dss['y']= xr.DataArray(y, dims=['num_lines', 'num_pixels'])
    dss = dss.set_coords(['x', 'y'])
    
    #GET A PERFECTLY REGULAR GRID FROM THE PREVIOUS LOCAL SWATH GRID
    # small sub-grid for scipy interp
    try :
        dl = 5*1000
        test_x = (dss.x > xd-dl) & (dss.x<xd+dl)
        ds_ = dss.where(test_x, drop=True)
        test_y = (ds_.y > yd-dl) & (ds_.y<yd+dl)
        ds_ = ds_.where(test_y, drop=True)
        
    except :# correct if the drifter point is two far outside the swath (must be corrected by a better selection of drifters under the swath)
        return pd.Series(np.zeros(len(variables)+2*len(ggrad_variables)), index = variables + ['ggradx_'+var for var in ggrad_variables]+['ggrady_'+var for var in ggrad_variables])
        
    try : 
        # coordinates of the regular grid
        dll =50
        box_x = np.arange(ds_.x.min()-dll, ds_.x.max()+dll, dll)
        box_y = np.arange(ds_.y.min()-dll, ds_.y.max()+dll, dll)
        # interpolate the interesting variable on this regular grid

        try:
            D = {}
            for var in total_variables : 
                Z = ds_[var].values.ravel()
                X = ds_.x.values.ravel()
                Y = ds_.y.values.ravel()
                _ds = xr.Dataset(data_vars=dict(x=(["a"], X), y=(["a"], Y), z=(["a"], Z)))
                _ds = _ds.dropna('a')
                #f = LinearNDInterpolator(list(zip(x, y)), z)
                #box_x, box_y = np.meshgrid(box_x, box_y) # for LinearNDInterpolator
                f = interp2d(_ds.x, _ds.y, _ds.z)
                Znew = f(box_x, box_y)
                da = xr.DataArray(Znew, coords={'x':box_x, 'y':box_y}, dims=[ 'y', 'x'], name=var)
                D[var]=da
        except :
            D = {}
            for var in total_variables : 
                lx = len(box_x)
                ly = len(box_y)
                da = xr.DataArray(np.zeros((ly, lx)), coords={'x':box_x, 'y':box_y}, dims=[ 'y', 'x'], name=var)
                D[var]=da
            print('pb with interp', cycle, swath, row_number)

     
    except : 
        D = {}
        for var in total_variables : 
            box_x = np.arange(43)
            box_y = np.arange(41)
            lx = 43
            ly = 41
            da = xr.DataArray(np.zeros((ly, lx)), coords={'x':box_x, 'y':box_y}, dims=[ 'y', 'x'], name=var)
            D[var]=da
        print('pb with ds_.min()', cycle, swath, row_number)
    
    
    # Compute g * gradient
    Dx = {'ggradx_'+var : g*D[var].differentiate('x') for var in ggrad_variables}
    Dy = {'ggrady_'+var : g*D[var].differentiate('y') for var in ggrad_variables}

    # Interpolate at the drifter position
    Dxd ={var : float(Dx[var].interp(x=xd, y=yd).values) for var in Dx}
    Dyd ={var : float(Dy[var].interp(x=xd, y=yd).values) for var in Dy}
    D ={var : float(D[var].interp(x=xd, y=yd).values) for var in variables}
    print(row_number)

    return pd.concat([pd.Series(D),pd.Series(Dxd),pd.Series(Dyd)])

ggrad_variables = ['cvl_mean_dynamic_topography_cnes_cls_22',
                 'cvl_mean_sea_surface_cnes_22_hybrid',
                 'cvl_ocean_tide_fes_2022',
                 'cvl_ssha_reference',
                 'duacs_ssha_karin_2_calibrated',
                 'duacs_ssha_karin_2_filtered',]

variables = ['duacs_relative_vorticity',
                    'duacs_speed_meridional',
                    'duacs_speed_meridional_abs',
                    'duacs_speed_zonal',
                    'duacs_speed_zonal_abs',
                   'cvl_swh_model', 
                   ]

def df_swot_interp_grad_one(dfr, variables, ggrad_variables):
    return swot_interp_grad_one(dfr.pass_number, dfr.cycle_number, dfr.longitude, dfr.latitude, dfr.row_number, variables, ggrad_variables)

def partition_swot_interp_grad(df) : 
    return df.apply(df_swot_interp_grad_one, variables=variables, ggrad_variables=ggrad_variables, axis=1, result_type='expand')
