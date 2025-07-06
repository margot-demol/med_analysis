import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

import matplotlib.pyplot as plt

import os
from glob import glob
from swot import browse_swot_250m, browse_swot_2km, add_grid_metrics,rotate_ggd
from cstes import swot_dir_2km, zarr_dir



# Stencil method
def apply_stencil_diff(image, var, dx, dy):
    """
    Apply same stencil as CLS
    input : 
        image : 2D array, image swot
        var : str, must be in 'cste', 'dx', 'dy', 'dxx', 'dyy', 'dxy'

        dx : float, x direction time step
        dy : float, y direction time step
    May be revised using Tranchant code : https://github.com/treden/SwotDiag/blob/main/SwotDiag/misc.py
    """
    from scipy.ndimage import convolve
    assert var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'], "var must be in 'dx', 'dy', 'dxx', 'dyy', 'dxy'"
    
    stx = np.array([[1/280, -4/105, 1/5, 4/5, 0, -4/5, -1/5, 4/105, -1/280]])
    sty = stx.T
    norm = -sum(np.arange(0, 9)*[1/280, -4/105, 1/5, 4/5, 0, -4/5, -1/5, 4/105, -1/280])

    if var == 'dx' : output = convolve(image, stx, mode='mirror')/dx/norm
    if var == 'dy' : output = convolve(image, sty, mode='mirror')/dy/norm
    if var == 'dxx' : output = convolve(convolve(image, stx, mode='mirror'), stx)/dx/dx/(norm**2)
    if var == 'dyy' : output = convolve(convolve(image, sty, mode='mirror'), sty)/dy/dy/(norm**2)
    if var == 'dxy' : output = convolve(convolve(image, stx, mode='mirror'), sty)/dy/dx/(norm**2)
    
    return xr.DataArray(output, dims=image.dims)

# Fitting kernel
def fitting_coeff(image, var, dx, dy): 
    """
    fitting method inspired from Tranchant
    input : 
        image : 2D array, image swot
        var : str, must be in 'cste', 'dx', 'dy', 'dxx', 'dyy', 'dxy'

        dx : float, x direction time step
        dy : float, y direction time step
    May be revised using Tranchant code : https://github.com/treden/SwotDiag/blob/main/SwotDiag/misc.py
    """
    #https://github.com/treden/SwotDiag/blob/main/SwotDiag/misc.py
    meaning_coeff = {'cste':0, 'dx':1, 'dy':2, 'dxx':3, 'dyy':4, 'dxy':5}
    assert var in meaning_coeff.keys(), "var must be in 'cste', 'dx', 'dy', 'dxx', 'dyy', 'dxy'"
    n=int(np.sqrt(image.shape[0]))
    if isinstance(image, xr.DataArray): image = image.values
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, copy=False)
    X, Y = np.meshgrid(x, y, copy=False)

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, Y**2, X*Y]).T

    coeff, r, rank, s = np.linalg.lstsq(A, image, rcond=-1)
    dxdy_diff = np.array([1, 1/dx, 1/dy, 1/dx/dx, 1/dy/dy, 1/dy/dx]).T
    coeff = coeff*dxdy_diff
    
    return coeff[meaning_coeff[var]]

def apply_fitting_kernel(swot_image, var, npts,dx, dy):
    """
    apply fitting method inspired from Tranchant
    input : 
        image : 2D array, image swot
        var : str, must be in 'cste', 'dx', 'dy', 'dxx', 'dyy', 'dxy'
        npts :  int, fitting kernel number of point
        dx : float, x direction time step
        dy : float, y direction time step
    May be revised using Tranchant code : https://github.com/treden/SwotDiag/blob/main/SwotDiag/misc.py
    """
    from scipy.ndimage import generic_filter
    assert npts%2 ==1, 'must be odd'
    return xr.DataArray(generic_filter(swot_image, fitting_coeff, size=[npts, npts], extra_keywords = {'var':var, 'dx':dx, 'dy':dy}), dims=swot_image.dims)

# Gaussian
def apply_gaussian_filter(swot_image, cutoff, mask, dx, dy):
    """
    Filter swot_image with a gaussian filter
    input : 
        swot_image : 2D array, image swot
        cutoff :  float, cutoff length (width at mid height of the gaussian filter)
        mask : 2D array with 0 out of swot swath and 1 within
        dx : float, x direction time step
        dy : float, y direction time step
    """
    from scipy.ndimage import convolve1d
    import scipy.signal.windows as wdw
    
    # Gaussian a la mano
    lambda_cutoff = cutoff/dx
    sigma_cutoff = lambda_cutoff * np.sqrt(np.log(2))
    Mg = int(2 * np.round(4*lambda_cutoff)+1)
    gaussian = wdw.gaussian(Mg, std = sigma_cutoff)
    fg = xr.DataArray(convolve1d(convolve1d(swot_image.fillna(0), gaussian, axis=0), gaussian, axis=1), dims=swot_image.dims)#unnromalized
    fm = xr.DataArray(convolve1d(convolve1d(mask, gaussian, axis=0), gaussian, axis=1), dims=mask.dims)#unnormalized
    return xr.DataArray(fg/fm, dims=swot_image.dims)#normalized

# Wraper
def define_eta(ds, mean_sla):
    """
     Compute calibrated and filtered DSL from SWOT product variables
     input : 
         ds :  swot xarray 
         mean_sla : boolean, if True, remove the mean SSH from the DSL
         
    """
    ds['etaf'] = ds.duacs_ssha_karin_2_filtered + ds.cvl_mean_dynamic_topography_cnes_cls_22 + ds.cvl_ocean_tide_fes_2022
    ds['etac'] = ds.duacs_ssha_karin_2_calibrated + ds.cvl_mean_dynamic_topography_cnes_cls_22 + ds.cvl_ocean_tide_fes_2022
    if mean_sla : 
        dsm = xr.open_dataset(os.path.join(zarr_dir, 'before_coloc','preprocessed_swot','swot2km', f'pass{int(ds.pass_number.mean().values)}_swot2km_meanssha.nc'))
        ds['etafm'] = ds.etaf - dsm.mean_sshaf
        ds['etacm'] = ds.etac - dsm.mean_sshac

def filter_diff_one(f, filter_diff_method = 'gaussian', filter_diff_kwargs = {'cutoff': 5e3}, mean_sla=False, distance_to_coast_min=None):
    """
    Filter or differenciate a SWOT image with one of the previous functions.
    input : 
        f : str, file containing SWOT image (netcdf)
        filter_diff_method : str, should be among: 'gaussian' (calls apply_gaussian_filter), 'fitting_kernel' (calls apply_fitting_kernel), 'diff_only' (call apply_stencil_diff) or xarray_diff (use xarray differentiate function)
        filter_diff_kwargs : dict, arguments of the chosen filter_diff_method
        mean_sla : boolean, if True, remove the mean SSH from the DSL
        distance_to_coast_min : select only points at more than XX km from the coast

    """
    ds = xr.open_dataset(f)
    define_eta(ds, mean_sla)
    ds = add_grid_metrics(ds)
    dx = float(ds.dx.mean())
    dy = float(ds.dy.mean())
    
    if distance_to_coast_min != None:
        ds = ds.where(ds.distance_to_coast>distance_to_coast_min, drop=True)

    vars_ = [v for v in list(ds.keys()) if ('etaf' in v)|('etac' in v)]+['duacs_editing_flag', 'longitude', 'latitude']
    
    ds =ds.drop_vars(['latitude_nadir','longitude_nadir','dx','phi','dy'])[vars_]
    
    if filter_diff_method == 'fitting_kernel':
        ds = ds.interpolate_na('num_pixels').interpolate_na('num_lines') # erreur dans l'entre-fauchée pour gaussian filter but ok for fitting kernel
    mask = ds.duacs_editing_flag.where(ds.duacs_editing_flag!=0,1).where(ds.duacs_editing_flag==0,0)
    try : 
        D = []
        for eta in [v for v in list(ds.keys()) if ('etaf' in v)|('etac' in v)] :
            swot_image = ds[eta]
            #gaussian method
            if filter_diff_method == 'gaussian' :
                assert list(filter_diff_kwargs.keys()) == ['cutoff'], 'The gaussian method needs one argument cutoff (gaussian x at H/2)'
                output = apply_gaussian_filter(swot_image,  **filter_diff_kwargs, mask = mask, dx=dx, dy=dy).rename('filtered_'+eta).to_dataset()

                #stencil diff
                #for var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'] : 
                #   output[var + '_'+eta] = apply_stencil_diff(output['filtered_'+eta], var, dx=dx, dy=dy)
                
                # xarray diff
                output['dx_'+eta] = output['filtered_'+eta].differentiate('num_pixels')/dx
                output['dy_'+eta] = output['filtered_'+eta].differentiate('num_lines')/dy
                output['dxx_'+eta] = output['dx_'+eta].differentiate('num_pixels')/dx
                output['dyy_'+eta] = output['dy_'+eta].differentiate('num_lines')/dy
                output['dxy_'+eta] = output['dx_'+eta].differentiate('num_lines')/dy
                D.append(output)
             
            # Fitting kernel method
            if filter_diff_method == 'fitting_kernel' :
                assert list(filter_diff_kwargs.keys()) == ['npts'], 'The fitting kernel method needs one argument npts (odd, kernel = [npts, npts])'
                output = apply_fitting_kernel(swot_image, **filter_diff_kwargs, var = 'cste', dx=dx, dy=dy).rename('filtered_'+eta).to_dataset()
                for var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'] : 
                    output[var + '_'+eta] = apply_fitting_kernel(swot_image, **filter_diff_kwargs, var=var, dx=dx, dy=dy)
                D.append(output)

            #diff only method
            if filter_diff_method == 'diff_only' :
                output = swot_image.rename(eta).to_dataset()
                for var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'] : 
                    output[var + '_'+eta] = apply_stencil_diff(swot_image, var, dx=dx, dy=dy)
                D.append(output)
                
            # xarray simple diff
            if filter_diff_method == 'xarray_diff' :
                output = swot_image.rename(eta).to_dataset()
                output['dx_'+eta] = output[eta].differentiate('num_pixels')/dx
                output['dy_'+eta] = output[eta].differentiate('num_lines')/dy
                output['dxx_'+eta] = output['dx_'+eta].differentiate('num_pixels')/dx
                output['dyy_'+eta] = output['dy_'+eta].differentiate('num_lines')/dy
                output['dxy_'+eta] = output['dx_'+eta].differentiate('num_lines')/dy
                D.append(output)
                    
    except: 
        assert False, f
    return xr.merge(D + [ds[['longitude', 'latitude']]]).where(ds.duacs_editing_flag==0)

def _concat(ds, filter_diff_method, filter_diff_kwargs, mean_sla, distance_to_coast_min):
    """
    Filter or differenciate some SWOT image with one of the previous functions.
    input : 
        ds : xarray dataset, contain SWOT file name and some properties
        filter_diff_method : str, should be among: 'gaussian' (calls apply_gaussian_filter), 'fitting_kernel' (calls apply_fitting_kernel), 'diff_only' (call apply_stencil_diff) or xarray_diff (use xarray differentiate function)
        filter_diff_kwargs : dict, arguments of the chosen filter_diff_method
        mean_sla : boolean, if True, remove the mean SSH from the DSL
        distance_to_coast_min : select only points at more than XX km from the coast

    """
    D = []
    for f in ds.file.values :
        D.append(filter_diff_one(f, filter_diff_method , filter_diff_kwargs, mean_sla = mean_sla, distance_to_coast_min=distance_to_coast_min))
    try : 
        xs = xr.concat(D, dim=ds.cycle_number)
    except : 
        assert False, f
    return xs

def compute_filter_diff(ds, filter_diff_method, filter_diff_kwargs, mean_sla, distance_to_coast_min=None):
    """
    Filter or differenciate all SWOT images with one of the previous functions with map_blocks.
    input : 
        ds : xarray dataset, contain SWOT file name and some properties
        filter_diff_method : str, should be among: 'gaussian' (calls apply_gaussian_filter), 'fitting_kernel' (calls apply_fitting_kernel), 'diff_only' (call apply_stencil_diff) or xarray_diff (use xarray differentiate function)
        filter_diff_kwargs : dict, arguments of the chosen filter_diff_method
        mean_sla : boolean, if True, remove the mean SSH from the DSL
        distance_to_coast_min : select only points at more than XX km from the coast

    """
    #template
    template = _concat(ds.isel(cycle_number = slice(0,2)), filter_diff_method ='gaussian', filter_diff_kwargs={'cutoff':5e3}, mean_sla=mean_sla, distance_to_coast_min=distance_to_coast_min).compute()
    template = template.isel(cycle_number=0).expand_dims({'cycle_number':ds.cycle_number}).chunk({**ds.chunks, **{'num_pixels':-1, 'num_lines':-1}})
    dims = [ "cycle_number", "num_lines", "num_pixels"]
    template = template.transpose(*dims)

    #perform the calculation
    ds_diff = ds.map_blocks(
        _concat,
        kwargs = dict( filter_diff_method =filter_diff_method, filter_diff_kwargs=filter_diff_kwargs, mean_sla=mean_sla, distance_to_coast_min=distance_to_coast_min),
        template=template,
    )
    return ds_diff
