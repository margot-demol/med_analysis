import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from numba import njit, prange

import matplotlib.pyplot as plt

import os
from glob import glob
from swot import browse_swot_250m, browse_swot_2km, add_grid_metrics, rotate_ggd
from cstes import swot_dir_2km, zarr_dir, version_swot

if version_swot == "1.0.2":
    sshaf_key = "duacs_ssha_karin_2_filtered"
    sshac_key = "duacs_ssha_karin_2_calibrated"
    mdt_key = "cvl_mean_dynamic_topography_cnes_cls_22"
    ocean_tide_key = "cvl_ocean_tide_fes_2022"

if version_swot == "2.0.1":
    sshaf_key = "duacs_ssha_karin_2_filtered"
    sshac_key = "duacs_ssha_karin_2_calibrated"
    mdt_key = "duacs_mean_dynamic_topography"
    oceantide_key = "cvl_ocean_tide_fes_2022"


# Stencil method


def apply_stencil_diff(image, var, dx, dy):
    """
    Apply same stencil as CLS = Arbic 2012
    input :
        image : 2D array, image swot
        var : str, must be in 'cste', 'dx', 'dy', 'dxx', 'dyy', 'dxy'

        dx : float, x direction time step
        dy : float, y direction time step
    May be revised using Tranchant code : https://github.com/treden/SwotDiag/blob/main/SwotDiag/misc.py
    """
    from scipy.ndimage import convolve

    assert var in [
        "dx",
        "dy",
        "dxx",
        "dyy",
        "dxy",
    ], "var must be in 'dx', 'dy', 'dxx', 'dyy', 'dxy'"

    S = -np.array([-3, 32, -168, 672, 0, -672, 168, -32, 3]) / 840
    norm = -sum(np.arange(0, 9) * S)
    stx = np.array([S / norm])
    sty = stx.T
    print(norm)

    if var == "dx":
        output = convolve(image, stx, mode="mirror") / dx
    if var == "dy":
        output = convolve(image, sty, mode="mirror") / dy
    if var == "dxx":
        output = (
            convolve(convolve(image, stx, mode="mirror"), stx) / dx / dx
        )  # /(norm**2)
    if var == "dyy":
        output = (
            convolve(convolve(image, sty, mode="mirror"), sty) / dy / dy
        )  # /(norm**2)
    if var == "dxy":
        output = (
            convolve(convolve(image, stx, mode="mirror"), sty) / dy / dx
        )  # /(norm**2)

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
    # https://github.com/treden/SwotDiag/blob/main/SwotDiag/misc.py
    meaning_coeff = {"cste": 0, "dx": 1, "dy": 2, "dxx": 3, "dyy": 4, "dxy": 5}
    assert (
        var in meaning_coeff.keys()
    ), "var must be in 'cste', 'dx', 'dy', 'dxx', 'dyy', 'dxy'"
    n = int(np.sqrt(image.shape[0]))
    if isinstance(image, xr.DataArray):
        image = image.values
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, copy=False)
    X, Y = np.meshgrid(x, y, copy=False)

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X * 0 + 1, X, Y, X**2, Y**2, X * Y]).T

    coeff, r, rank, s = np.linalg.lstsq(A, image, rcond=-1)
    dxdy_diff = np.array([1, 1 / dx, 1 / dy, 1 / dx / dx, 1 / dy / dy, 1 / dy / dx]).T
    coeff = coeff * dxdy_diff

    return coeff[meaning_coeff[var]]


def apply_fitting_kernel(swot_image, var, npts, dx, dy):
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

    assert npts % 2 == 1, "must be odd"
    return xr.DataArray(
        generic_filter(
            swot_image,
            fitting_coeff,
            size=[npts, npts],
            extra_keywords={"var": var, "dx": dx, "dy": dy},
        ),
        dims=swot_image.dims,
    )


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
    lambda_cutoff = cutoff / dx
    sigma_cutoff = lambda_cutoff * np.sqrt(np.log(2))
    Mg = int(2 * np.round(4 * lambda_cutoff) + 1)
    gaussian = wdw.gaussian(Mg, std=sigma_cutoff)
    fg = xr.DataArray(
        convolve1d(
            convolve1d(swot_image.fillna(0), gaussian, axis=0), gaussian, axis=1
        ),
        dims=swot_image.dims,
    )  # unnromalized
    fm = xr.DataArray(
        convolve1d(convolve1d(mask, gaussian, axis=0), gaussian, axis=1), dims=mask.dims
    )  # unnormalized
    return xr.DataArray(fg / fm, dims=swot_image.dims)  # normalized


# Wraper
def define_eta(ds, mean_sla):
    """
    Compute calibrated and filtered DSL from SWOT product variables
    input :
        ds :  swot xarray
        mean_sla : boolean, if True, remove the mean SSH from the DSL

    """
    ds["etaf"] = (
        ds[sshaf_key] + ds[mdt_key]
    )  # + ds[oceantide_key] #not necessary in the Western Mediterranean Sea
    ds["etac"] = ds[sshac_key] + ds[mdt_key]  # + ds[oceantide_key]
    if mean_sla:
        dsm = xr.open_dataset(
            os.path.join(
                zarr_dir,
                "before_coloc",
                "preprocessed_swot",
                "swot2km",
                f"pass{int(ds.pass_number.mean().values)}_swot2km_meanssha.nc",
            )
        )
        ds["etafm"] = ds.etaf - dsm.mean_sshaf
        ds["etacm"] = ds.etac - dsm.mean_sshac


def filter_diff_one(
    f,
    filter_diff_method="gaussian",
    filter_diff_kwargs={"cutoff": 5e3},
    mean_sla=False,
    distance_to_coast_min=None,
):
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
    # correct SWOT product time
    if version_swot == "2.0.1":
        ds["time"] = ds.time - np.timedelta64(946684800000000000, "ns")

    define_eta(ds, mean_sla)
    ds = add_grid_metrics(ds)
    dx = np.round(float(ds.dx.mean()))
    dy = np.round(float(ds.dy.mean()))

    if distance_to_coast_min != None:
        ds = ds.where(ds.distance_to_coast > distance_to_coast_min, drop=True)

    vars_ = [v for v in list(ds.keys()) if ("etaf" in v) | ("etac" in v)] + [
        "duacs_editing_flag",
        "longitude",
        "latitude",
    ]

    ds = ds.drop_vars(["latitude_nadir", "longitude_nadir", "dx", "phi", "dy"])[vars_]

    if filter_diff_method == "fitting_kernel":
        ds = ds.interpolate_na("num_pixels").interpolate_na(
            "num_lines"
        )  # erreur dans l'entre-fauchée pour gaussian filter but ok for fitting kernel
    # mask = ds.duacs_editing_flag.where(ds.duacs_editing_flag!=0,1).where(ds.duacs_editing_flag==0,0)
    mask = xr.where(
        (~np.isnan(ds.longitude)) & (~np.isnan(ds.latitude)) & (~np.isnan(ds["etaf"])),
        1,
        0,
    )

    try:
        D = []
        for eta in [v for v in list(ds.keys()) if ("etaf" in v) | ("etac" in v)]:
            swot_image = ds[eta]
            # gaussian method (old version apply 2 1D gaussian filter succesively)
            if filter_diff_method == "gaussian_2times1D":
                assert list(filter_diff_kwargs.keys()) == [
                    "cutoff"
                ], "The gaussian method needs one argument cutoff (gaussian x at H/2)"
                output = (
                    apply_gaussian_filter(
                        swot_image, **filter_diff_kwargs, mask=mask, dx=dx, dy=dy
                    )
                    .rename("filtered_" + eta)
                    .to_dataset()
                )

                # stencil diff
                # for var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'] :
                #   output[var + '_'+eta] = apply_stencil_diff(output['filtered_'+eta], var, dx=dx, dy=dy)

                # xarray diff
                output["dx_" + eta] = (
                    output["filtered_" + eta].differentiate("num_pixels") / dx
                )
                output["dy_" + eta] = (
                    output["filtered_" + eta].differentiate("num_lines") / dy
                )
                output["dxx_" + eta] = (
                    output["dx_" + eta].differentiate("num_pixels") / dx
                )
                output["dyy_" + eta] = (
                    output["dy_" + eta].differentiate("num_lines") / dy
                )
                output["dxy_" + eta] = (
                    output["dx_" + eta].differentiate("num_lines") / dy
                )
                D.append(output)

            # gaussian method with 2D filter (version 'a la mano' with numba by aurelien)
            elif filter_diff_method == "gaussian_aviso":
                assert list(filter_diff_kwargs.keys()) == [
                    "cutoff"
                ], "The gaussian method needs one argument cutoff (gaussian x at H/2)"

                # AVISO
                L4_key = "L4_withnadirswot"
                aviso = xr.open_dataset(
                    os.path.join(
                        zarr_dir, "before_coloc", "L4_sealevel", L4_key + ".nc"
                    )
                )
                # select only area where swath is ... adjust for swath 16
                aviso = aviso.sel(
                    longitude=slice(ds.longitude.min() - 1, ds.longitude.max() + 1),
                    latitude=slice(ds.latitude.min() - 1, ds.latitude.max() + 1),
                )
                aviso = aviso.interp(time=ds.time.mean().compute().values)
                aviso = aviso["adt"]
                # return swot_image, mask, dx, aviso

                output = (
                    apply_gaussian_filter_aviso(
                        swot_image,
                        **filter_diff_kwargs,
                        mask=mask,
                        dx=dx,
                        aviso_image=aviso,
                    )
                    .rename("filtered_" + eta)
                    .to_dataset()
                )

                # stencil diff
                # for var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'] :
                #   output[var + '_'+eta] = apply_stencil_diff(output['filtered_'+eta], var, dx=dx, dy=dy)

                # xarray diff
                output["dx_" + eta] = (
                    output["filtered_" + eta].differentiate("num_pixels") / dx
                )
                output["dy_" + eta] = (
                    output["filtered_" + eta].differentiate("num_lines") / dy
                )
                output["dxx_" + eta] = (
                    output["dx_" + eta].differentiate("num_pixels") / dx
                )
                output["dyy_" + eta] = (
                    output["dy_" + eta].differentiate("num_lines") / dy
                )
                output["dxy_" + eta] = (
                    output["dx_" + eta].differentiate("num_lines") / dy
                )
                D.append(output)

            # gaussian method with 2D filter (version 'a la mano' with numba by aurelien)
            # elif filter_diff_method == 'gaussian_aviso' :
            #    assert list(filter_diff_kwargs.keys()) == ['cutoff'], 'The gaussian method needs one argument cutoff (gaussian x at H/2)'

            #    #AVISO
            #    aviso = xr.open_dataset(os.path.join(zarr_dir, 'before_coloc', 'L4_sealevel', L4_key+'.nc'))
            #    # select only area where swath is ... adjust for swath 16
            #    aviso = aviso.sel(longitude=slice(ds.longitude.min()-1,ds.longitude.max()+1), latitude=slice(ds.latitude.min()-1, ds.latitude.max()+1))
            #    aviso = aviso.interp(time = ds.time.mean().compute().values)
            #    aviso = aviso['adt']
            # return swot_image, mask, dx, aviso

            #    output = apply_gaussian_filter_nb(swot_image, **filter_diff_kwargs, mask = mask, dx = dx).rename('filtered_'+eta).to_dataset()

            #   #stencil diff
            #    #for var in ['dx', 'dy', 'dxx', 'dyy', 'dxy'] :
            #    #   output[var + '_'+eta] = apply_stencil_diff(output['filtered_'+eta], var, dx=dx, dy=dy)

            #    # xarray diff
            #    output['dx_'+eta] = output['filtered_'+eta].differentiate('num_pixels')/dx
            #    output['dy_'+eta] = output['filtered_'+eta].differentiate('num_lines')/dy
            #    output['dxx_'+eta] = output['dx_'+eta].differentiate('num_pixels')/dx
            #    output['dyy_'+eta] = output['dy_'+eta].differentiate('num_lines')/dy
            #    output['dxy_'+eta] = output['dx_'+eta].differentiate('num_lines')/dy
            #    D.append(output)

            # Fitting kernel method
            elif filter_diff_method == "fitting_kernel":
                assert list(filter_diff_kwargs.keys()) == [
                    "npts"
                ], "The fitting kernel method needs one argument npts (odd, kernel = [npts, npts])"
                output = (
                    apply_fitting_kernel(
                        swot_image, **filter_diff_kwargs, var="cste", dx=dx, dy=dy
                    )
                    .rename("filtered_" + eta)
                    .to_dataset()
                )
                for var in ["dx", "dy", "dxx", "dyy", "dxy"]:
                    output[var + "_" + eta] = apply_fitting_kernel(
                        swot_image, **filter_diff_kwargs, var=var, dx=dx, dy=dy
                    )
                D.append(output)

            # diff only method
            elif filter_diff_method == "diff_only":
                output = swot_image.rename(eta).to_dataset()
                for var in ["dx", "dy", "dxx", "dyy", "dxy"]:
                    output[var + "_" + eta] = apply_stencil_diff(
                        swot_image, var, dx=dx, dy=dy
                    )
                D.append(output)

            # xarray simple diff
            elif filter_diff_method == "xarray_diff":
                output = swot_image.rename(eta).to_dataset()
                output["dx_" + eta] = output[eta].differentiate("num_pixels") / dx
                output["dy_" + eta] = output[eta].differentiate("num_lines") / dy
                output["dxx_" + eta] = (
                    output["dx_" + eta].differentiate("num_pixels") / dx
                )
                output["dyy_" + eta] = (
                    output["dy_" + eta].differentiate("num_lines") / dy
                )
                output["dxy_" + eta] = (
                    output["dx_" + eta].differentiate("num_lines") / dy
                )
                D.append(output)

    except:
        assert False, f
    return xr.merge(D + [ds[["longitude", "latitude"]]]).where(
        ds.duacs_editing_flag == 0
    )


def _concat(
    ds, filter_diff_method, filter_diff_kwargs, mean_sla, distance_to_coast_min
):
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
    for f in ds.file.values:
        D.append(
            filter_diff_one(
                f,
                filter_diff_method,
                filter_diff_kwargs,
                mean_sla=mean_sla,
                distance_to_coast_min=distance_to_coast_min,
            )
        )
    try:
        xs = xr.concat(D, dim=ds.cycle_number)
    except:
        assert False, f
    return xs


def compute_filter_diff(
    ds, filter_diff_method, filter_diff_kwargs, mean_sla, distance_to_coast_min=None
):
    """
    Filter or differenciate all SWOT images with one of the previous functions with map_blocks.
    input :
        ds : xarray dataset, contain SWOT file name and some properties
        filter_diff_method : str, should be among: 'gaussian' (calls apply_gaussian_filter), 'fitting_kernel' (calls apply_fitting_kernel), 'diff_only' (call apply_stencil_diff) or xarray_diff (use xarray differentiate function)
        filter_diff_kwargs : dict, arguments of the chosen filter_diff_method
        mean_sla : boolean, if True, remove the mean SSH from the DSL
        distance_to_coast_min : select only points at more than XX km from the coast

    """
    # template
    template = _concat(
        ds.isel(cycle_number=slice(0, 2)),
        filter_diff_method="gaussian",
        filter_diff_kwargs={"cutoff": 5e3},
        mean_sla=mean_sla,
        distance_to_coast_min=distance_to_coast_min,
    ).compute()
    template = (
        template.isel(cycle_number=0)
        .expand_dims({"cycle_number": ds.cycle_number})
        .chunk({**ds.chunks, **{"num_pixels": -1, "num_lines": -1}})
    )
    dims = ["cycle_number", "num_lines", "num_pixels"]
    template = template.transpose(*dims)

    # perform the calculation
    ds_diff = ds.map_blocks(
        _concat,
        kwargs=dict(
            filter_diff_method=filter_diff_method,
            filter_diff_kwargs=filter_diff_kwargs,
            mean_sla=mean_sla,
            distance_to_coast_min=distance_to_coast_min,
        ),
        template=template,
    )
    return ds_diff


## Gaussian filtering - Aurelien ###


@njit()
def _dist_geo(lon1, lat1, lon2, lat2):
    """Computes the Haversine distance in kilometres between two points
    :param x: first point or points as array, each as array of latitude, longitude in degrees
    :param y: second point or points as array, each as array of latitude, longitude in degrees
    :return: distance between the two points in kilometres
    """
    deg2rad = np.pi / 180.0
    llat1 = lat1 * deg2rad
    llat2 = lat2 * deg2rad
    llon1 = lon1 * deg2rad
    llon2 = lon2 * deg2rad
    arclen = 2 * np.arcsin(
        np.sqrt(
            (np.sin((llat2 - llat1) / 2)) ** 2
            + np.cos(llat1) * np.cos(llat2) * (np.sin((llon2 - llon1) / 2)) ** 2
        )
    )
    earth_radius = 6370e3  # approximate Earth radius at 40degN in meters
    # https://en.wikipedia.org/wiki/Earth_radius
    return arclen * earth_radius


@njit()
def _gauss2d(d, sigma):
    sigma2 = sigma**2
    d2 = d**2
    # return (0.5/np.pi/sigma2) * np.exp(-0.5/sigma2*d**2)
    return np.exp(-0.5 * d2 / sigma2)  # shortcut expecting normalization down the line


@njit(parallel=True)
def nb_filter_gaussian(
    lon0,
    lat0,
    v0,
    mask0,
    sigma,
    truncate,
):
    """filter data with Gaussian filter"""
    vf = np.zeros_like(v0)
    weight0 = np.zeros_like(v0)
    n0 = len(vf)

    sigma_x_truncate = sigma * truncate

    for i0 in prange(n0):
        # loop over primary data points
        for j0 in range(n0):
            d = _dist_geo(lon0[i0], lat0[i0], lon0[j0], lat0[j0])
            if mask0[j0] > 0 and d < sigma_x_truncate:
                w = _gauss2d(d, sigma)
                weight0[i0] += w
                vf[i0] += v0[j0] * w
        vf[i0] = vf[i0] / weight0[i0]

    return vf, weight0


@njit(parallel=True)
def nb_filter_gaussian_combined(
    lon0,
    lat0,
    v0,
    mask0,
    lon1,
    lat1,
    v1,
    area_ratio,
    sigma,
    truncate,
):
    """filter data with Gaussian filter leveraging external data for points where the primary
    data source has no nearby points"""
    vf = np.zeros_like(v0)
    weight0 = np.zeros_like(v0)
    weight1 = np.zeros_like(v0)
    n0 = len(vf)
    n1 = len(v1)

    sigma_x_truncate = sigma * truncate

    for i0 in prange(n0):
        # loop over primary data points
        for j0 in range(n0):
            d = _dist_geo(lon0[i0], lat0[i0], lon0[j0], lat0[j0])
            if mask0[j0] > 0 and d < sigma_x_truncate:
                w = _gauss2d(d, sigma)
                weight0[i0] += w
                vf[i0] += v0[j0] * w
        # loop over secondary data points
        for i1 in range(n1):
            d = _dist_geo(lon0[i0], lat0[i0], lon1[i1], lat1[i1])
            if d < sigma_x_truncate:
                w = _gauss2d(d, sigma) * area_ratio
                weight1[i0] += w
                vf[i0] += v1[i1] * w

        vf[i0] = vf[i0] / (weight0[i0] + weight1[i0])

    return vf, weight0, weight1


@njit(parallel=True)
def nb_mask_grid(lon, lat, lon1, lat1, dl):
    """mask secondary data if close to primary one"""
    n = len(lon)
    n1 = len(lon1)
    mask1 = np.ones_like(lon1)

    for i1 in prange(n1):
        for i in range(n):
            d = _dist_geo(lon[i], lat[i], lon1[i1], lat1[i1])
            if d < dl:
                mask1[i1] = 0
                break

    return mask1


def apply_gaussian_filter_nb(
    swot_image,
    cutoff,
    mask,
    dx,
):
    """
    Filter swot_image with a gaussian filter with numba
    input :
        swot_image : 2D array, image swot
        cutoff :  float, cutoff length (width at mid height of the gaussian filter) in m
        mask : 2D array with 0 out of swot swath and 1 within
        dx : float, x direction grid step in m
    """

    # try being consistent with Margot's initial choice
    sigma = cutoff * np.sqrt(np.log(2))
    truncate = 4.0 * cutoff / sigma

    # stack fields
    ds = xr.merge([swot_image.rename("v"), mask.rename("mask")]).stack(
        point=["num_lines", "num_pixels"]
    )
    swot_stacked = ds
    ds = ds.where(ds["mask"]).fillna(0.0)
    lon0 = ds["longitude"].data
    lat0 = ds["latitude"].data
    v0 = ds["v"].data
    mask0 = ds["mask"].data

    # filter
    vf, weight0 = nb_filter_gaussian(
        lon0,
        lat0,
        v0,
        mask0,
        sigma,
        truncate,
    )
    swot_stacked["vf"] = ("point", vf)
    swot_stacked = swot_stacked.assign_coords(
        weight0=("point", weight0),
    )

    return swot_stacked["vf"].unstack()


def apply_gaussian_filter_aviso(
    swot_image,
    cutoff,
    mask,
    dx,
    aviso_image,
):
    """
    Filter swot_image with a gaussian filter and aviso outside tracks
    input :
        swot_image : 2D array, image swot
        cutoff :  float, cutoff length (width at mid height of the gaussian filter) in m
        mask : 2D array with 0 out of swot swath and 1 within
        dx : float, x direction grid step in m
        aviso_image: xr.DataArray
    """

    # try being consistent with Margot's initial choice
    sigma = cutoff * np.sqrt(np.log(2))
    truncate = 4.0 * cutoff / sigma

    # compute relative weights
    dlon = float(aviso_image.longitude.diff("longitude").median())
    dlat = float(aviso_image.latitude.diff("latitude").median())
    _lat = float(aviso_image.latitude.median())
    unit_area_aviso = float(dlon * np.cos(np.deg2rad(_lat)) * dlat * (111e3) ** 2)

    unit_area_swot = dx**2
    area_ratio = unit_area_aviso / unit_area_swot

    # stack swot fields
    ds = xr.merge([swot_image.rename("v"), mask.rename("mask")]).stack(
        point=["num_lines", "num_pixels"]
    )
    swot_stacked = ds
    ds = ds.where(ds["mask"]).fillna(0.0)
    lon0, lat0 = ds["longitude"].data, ds["latitude"].data
    v0 = ds["v"].data
    mask0 = ds["mask"].data

    # mask and stack aviso fields
    da = aviso_image.squeeze().stack(point=["longitude", "latitude"])
    lon1, lat1 = da.longitude.data, da.latitude.data
    da = da.assign_coords(
        mask0=xr.where(np.isnan(da), 0, 1),
        mask1=("point", nb_mask_grid(lon0, lat0, lon1, lat1, dx)),
    )
    da = da.where((da.mask0 * da.mask1) > 0, drop=True)
    lon1, lat1 = da.longitude.data, da.latitude.data
    v1 = da.data

    # filter
    vf, weight0, weight1 = nb_filter_gaussian_combined(
        lon0,
        lat0,
        v0,
        mask0,
        lon1,
        lat1,
        v1,
        area_ratio,
        sigma,
        truncate,
    )
    swot_stacked["vf"] = ("point", vf)
    swot_stacked = swot_stacked.assign_coords(
        weight0=("point", weight0),
        weight1=("point", weight1),
    )

    return swot_stacked["vf"].unstack()
