import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import os
from glob import glob

from cstes import c0, U2, zarr_dir, surface_drifters, depth_drifters, depth_100, depth_50


""" CREATE DATASET """

drifters_sources = 'all_med_variational_10min_v0.nc'

drifter_file = os.path.join(zarr_dir, 'drifters_'+drifters_sources.replace('.nc', '.csv'))
DRIFTER = {'nofilter' : os.path.join(zarr_dir, 'drifters_'+drifters_sources.replace('.nc', '.csv')),
           '0.5cpd' : os.path.join(zarr_dir, 'drifters_'+drifters_sources.replace('.nc', '_filtered05.csv')),
           '1.5cpd' : os.path.join(zarr_dir, 'drifters_'+drifters_sources.replace('.nc', '_filtered15.csv')),
           '2.5cpd' : os.path.join(zarr_dir, 'drifters_'+drifters_sources.replace('.nc', '_filtered25.csv')),
           '3cpd' : os.path.join(zarr_dir, 'drifters_'+drifters_sources.replace('.nc', '_filtered3.csv')),
          }
ALTI = {
    'swot250naive' : os.path.join(zarr_dir, 'swot_250_'+drifters_sources.replace('.nc', '')+'_naive.csv'), 
    'swot250gauss1e3' : os.path.join(zarr_dir, 'swot_250_'+drifters_sources.replace('.nc', '')+'_gauss1e3.csv'), 
    'swot250gauss2e3' : os.path.join(zarr_dir, 'swot_250_'+drifters_sources.replace('.nc', '')+'_gauss2e3.csv'),
    'swot250gauss3e3' : os.path.join(zarr_dir, 'swot_250_'+drifters_sources.replace('.nc', '')+'_gauss3e3.csv'), 
    'swot250gauss4e3' : os.path.join(zarr_dir, 'swot_250_'+drifters_sources.replace('.nc', '')+'_gauss4e3.csv'), 
    'swot250gauss5e3' : os.path.join(zarr_dir, 'swot_250_'+drifters_sources.replace('.nc', '')+'_gauss5e3.csv'), 
    'swot2kmnaive' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_naive.csv'), 
    'swot2kmgauss1e3' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_gauss1e3.csv'), 
    'swot2kmgauss2e3' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_gauss2e3.csv'),
    'swot2kmgauss3e3' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_gauss3e3.csv'), 
    'swot2kmgauss4e3' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_gauss4e3.csv'), 
    'swot2kmgauss5e3' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_gauss5e3.csv'), 
    'swot2kmgauss1e4' : os.path.join(zarr_dir, 'swot_2km_'+drifters_sources.replace('.nc', '')+'_gauss1e4.csv'), 
}
WD = {'era5' : os.path.join(zarr_dir, 'era5_'+drifters_sources.replace('.nc', '')+'.csv')}

dtypes = {'drifter_id':str}

def prepared_drifters(drifter_key) : 
    dfr = pd.read_csv(DRIFTER[drifter_key], parse_dates=['datetime'], dtype=dtypes).set_index('row_number')
    dfr['f'] =  2 * 2 * np.pi / 86164.1 * np.sin(dfr.latitude * np.pi / 180)
    dfr['core']= -dfr.f * dfr.velocity_north
    dfr['corn']= dfr.f * dfr.velocity_east
    dfr = dfr.rename(columns = {'acceleration_north':'accn', 'acceleration_east':'acce'})
    return dfr[['datetime', 'longitude', 'latitude', 'pass_number','time_to_swot','cycle_number','cycle_date','drifter_id','drifter_type','accn', 'acce', 'core', 'corn']]

def prepared_alti(alti_key, ggd_var=['duacs_ssha_karin_2_filtered','duacs_ssha_karin_2_calibrated','cvl_mean_dynamic_topography_cnes_cls_22','cvl_ocean_tide_fes_2022']) :
    dfs = pd.read_csv(ALTI[alti_key]).set_index('row_number')
    dfs['f'] =  2 * 2 * np.pi / 86164.1 * np.sin(dfs.latitude * np.pi / 180)
    dfs['ggde_fromduacsv'] = dfs.f * dfs.duacs_speed_meridional_abs
    dfs['ggdn_fromduacsv'] = -dfs.f *dfs.duacs_speed_zonal_abs
    dfs = dfs[['ggde_'+v for v in ggd_var]+['ggdn_'+v for v in ggd_var]+['ggde_fromduacsv', 'ggdn_fromduacsv']]
    dfs['ggde_etaf'] = dfs.ggde_duacs_ssha_karin_2_filtered + dfs.ggde_cvl_mean_dynamic_topography_cnes_cls_22+dfs.ggde_cvl_ocean_tide_fes_2022
    dfs['ggdn_etaf'] = dfs.ggdn_duacs_ssha_karin_2_filtered + dfs.ggdn_cvl_mean_dynamic_topography_cnes_cls_22+dfs.ggdn_cvl_ocean_tide_fes_2022
    dfs['ggde_adtf'] = dfs.ggde_duacs_ssha_karin_2_filtered + dfs.ggde_cvl_mean_dynamic_topography_cnes_cls_22
    dfs['ggdn_adtf'] = dfs.ggdn_duacs_ssha_karin_2_filtered + dfs.ggdn_cvl_mean_dynamic_topography_cnes_cls_22
    dfs['ggde_etac'] = dfs.ggde_duacs_ssha_karin_2_calibrated + dfs.ggde_cvl_mean_dynamic_topography_cnes_cls_22+dfs.ggde_cvl_ocean_tide_fes_2022
    dfs['ggdn_etac'] = dfs.ggdn_duacs_ssha_karin_2_calibrated + dfs.ggdn_cvl_mean_dynamic_topography_cnes_cls_22+dfs.ggdn_cvl_ocean_tide_fes_2022
    dfs['ggde_adtc'] = dfs.ggde_duacs_ssha_karin_2_calibrated + dfs.ggde_cvl_mean_dynamic_topography_cnes_cls_22
    dfs['ggdn_adtc'] = dfs.ggdn_duacs_ssha_karin_2_calibrated + dfs.ggdn_cvl_mean_dynamic_topography_cnes_cls_22
    return dfs#.rename(columns ={v : v+'_' +alti_key for v in dfs})
    
def prepared_wd(wd_key): 
    dfw = pd.read_csv(WD[wd_key]).set_index('row_number')
    dfw = dfw[[v for v in dfw if ('wde' in v or 'wdn' in v)]]
    return dfw#.rename(columns ={v : v+'_' +wd_key for v in dfw})
    
def create_id_comb(drifter_key, alti_key, wd_key = 'era5', ggd_var='etaf', wd_depth='0'):
    return drifter_key + '__' + alti_key +'_' + ggd_var + '__' + wd_depth + wd_key

def select_nearest_swot_coloc(df):
    idx = df.groupby(['pass_number','cycle_number','drifter_id']).time_to_swot.idxmin()
    return df.loc[idx]

def one_coloc(drifter_key, alti_key, wd_key = 'era5', ggd_var='etaf', wd_depth='0'):
    df = pd.concat([prepared_drifters(drifter_key),
                    prepared_alti(alti_key)[['ggde_'+ggd_var,'ggdn_'+ggd_var]].rename(columns = {'ggde_'+ggd_var:'ggde','ggdn_'+ggd_var:'ggdn'}), 
                    prepared_wd(wd_key)[['wde'+wd_depth,'wdn'+wd_depth]].rename(columns = {'wde'+wd_depth : 'wde','wdn'+wd_depth:'wdn'}),
                   ], axis=1).dropna() # with dropna, depends on the altimetry filter
    id_comb = create_id_comb(drifter_key, alti_key, wd_key, ggd_var, wd_depth)
    
    #depth
    depth = df['pass_number'].copy()
    depth.loc[df['drifter_type'].isin(surface_drifters)]=0
    depth.loc[df['drifter_type'].isin(depth_drifters)]=15
    depth.loc[df['drifter_id'].isin(depth_50)]=50
    depth.loc[df['drifter_id'].isin(depth_100)]=100

    df['depth'] = depth
    print(id_comb)
    
    return df, id_comb

        
def dataset_coloc_combs(comb_list, nearest =True):
    D = []
    for comb in comb_list :
        df, id_comb = one_coloc(**comb)

        if nearest :
            # select only nearest in time colocalisation
            df = select_nearest_swot_coloc(df)#.dropna()
            
        coords_list = ['datetime', 'longitude', 'latitude', 'pass_number','time_to_swot','cycle_number','drifter_id', 'drifter_type','depth']
        ds = df[coords_list].to_xarray()
        ds['id_comb'] = id_comb
        
        for direction in ['e', 'n'] : 
            if direction == 'e' : l = {'acc':'acce', 'cor':'core', 'ggd':'ggde', 'wd':'wde'}
            if direction == 'n' : l = {'acc':'accn', 'cor':'corn', 'ggd':'ggdn', 'wd':'wdn'}
                
            # sum 
            ds['sum'+direction] = sum([df[l[v]] for v in l])

            # simple var + sum- one term
            for v in l :
                ds[l[v]] = df[l[v]]
                ds['exc'+direction+'_'+v] = ds['sum'+direction]-ds[l[v]]
                
            #2 by 2 product    
            import itertools  
            couple = list(itertools.combinations(list(l.keys()),2))
            for c in couple : 
                ds['prod'+direction+'_'+'_'.join(c)] = ds[c[0]+direction] * ds[c[1]+direction]
                
        D.append(ds.set_coords('id_comb'))
    return xr.concat(D, dim='id_comb').set_coords(coords_list)

""" PLOTS """

def synthetic_figure(df, ax, xlim=None, aviso=False, dir = 'e'):
    from cstes import U2, c0

    plt.rcParams["axes.edgecolor"] = "w"
    a = 1.5
    bbox = dict(facecolor="w", alpha=0.8, edgecolor="w")

    ts = df["sigma"+dir]
    print(ts)
    # gap between bars for readability
    if xlim:
        b = xlim / 400
    else:
        b = ts / 400

    # b = 1e-10

    ## INDIVIDUAL MS ##
    ax.barh(2 * a, df["ACC"+dir], color=c0["acc"], label="Lagrangian acceleration")
    ax.barh(
        2 * a,
        df["COR"+dir],
        left=df["ACC"+dir] + b,
        color=c0["cor"],
        label="Coriolis acceleration",
    )
    ax.barh(
        2 * a,
        df["GGD"+dir],
        left=df["ACC"+dir] + df["COR"+dir] + 2 * b,
        color=c0["ggd"],
        label="Pressure gradient term",
    )
    ax.barh(
        2 * a,
        df["WD"+dir],
        left=df["ACC"+dir] + df["COR"+dir] + df["GGD"+dir] + 3 * b,
        color=c0["wd"],
        label="Wind term",
    )

    ax.text(ts / 2, 2 * a + 0.5, r"Individual MS $A_i$", ha="center")
    # percentage + MS
    key = ["ACC"+dir, "COR"+dir, "GGD"+dir, "WD"+dir]
    for i in range(len(key)):
        ax.text(
            sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b,
            2 * a,
            f"{int(np.rint((df[key[i]]/ts)*100))} %",
            ha="center",
            bbox=bbox,
        )
        ax.text(
            sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b,
            2 * a - 0.55,
            f"{np.round(df[key[i]],2)}",
            ha="center",
        )

    # accolade
    c = 1e-12 * U2
    id1 = 0
    id2 = ts + 3 * b
    bx = [id1, id1, id2, id2]
    by = [3.70, 3.75, 3.75, 3.70]
    ax.plot(bx, by, "k-", lw=2)
    ax.text(ts, 3.8, r"$\Sigma$", fontsize=15, ha="center")

    ## CAPTURED PHYSICAL + ERRORS PARTS ##
    plt.rcParams["hatch.linewidth"] = 8
    plt.rcParams["hatch.color"] = "lightgrey"
    ax.barh(1 * a, df["B"+dir +"_acc"], color=c0["acc"])
    ax.barh(1 * a, df["E"+dir +"_acc"], left=df["B"+dir +"_acc"], color=c0["acc"], hatch="/")
    ax.barh(1 * a, df["B"+dir +"_cor"], left=df["B"+dir +"_acc"] + df["E"+dir +"_acc"] + b, color=c0["cor"])
    ax.barh(
        1 * a,
        df["E"+dir +"_cor"],
        left=df["B"+dir +"_acc"] + df["E"+dir +"_acc"] + df["B"+dir +"_cor"],
        color=c0["cor"],
        hatch="/",
    )
    ax.barh(
        1 * a,
        df["B"+dir +"_ggd"],
        left=df["B"+dir +"_acc"] + df["E"+dir +"_acc"] + df["B"+dir +"_cor"] + df["E"+dir +"_cor"] + b,
        color=c0["ggd"],
    )
    if df["E"+dir +"_ggd"] > 0:
        ax.barh(
            1 * a,
            df["E"+dir +"_ggd"],
            left=df["B"+dir +"_acc"] + df["E"+dir +"_acc"] + df["B"+dir +"_cor"] + df["E"+dir +"_cor"] + df["B"+dir +"_ggd"],
            color=c0["ggd"],
            hatch="/",
        )
        ax.barh(
            1 * a,
            df["B"+dir +"_wd"],
            left=df["B"+dir +"_acc"]
            + df["E"+dir +"_acc"]
            + df["B"+dir +"_cor"]
            + df["E"+dir +"_cor"]
            + df["B"+dir +"_ggd"]
            + df["E"+dir +"_ggd"]
            + b,
            color=c0["wd"],
        )
        ax.barh(
            1 * a,
            df["E"+dir +"_wd"],
            left=df["B"+dir +"_acc"]
            + df["E"+dir +"_acc"]
            + df["B"+dir +"_cor"]
            + df["E"+dir +"_cor"]
            + df["B"+dir +"_ggd"]
            + df["E"+dir +"_ggd"]
            + df["B"+dir +"_wd"],
            color=c0["wd"],
            hatch="/",
        )
    else:
        ax.barh(
            1 * a,
            df["B"+dir +"_wd"],
            left=df["B"+dir +"_acc"]
            + df["E"+dir +"_acc"]
            + df["B"+dir +"_cor"]
            + df["E"+dir +"_cor"]
            + df["B"+dir +"_ggd"]
            + 2 * b,
            color=c0["wd"],
        )
        ax.barh(
            1 * a,
            df["E"+dir +"_wd"],
            left=df["B"+dir +"_acc"]
            + df["E"+dir +"_acc"]
            + df["B"+dir +"_cor"]
            + df["E"+dir +"_cor"]
            + df["B"+dir +"_ggd"]
            + df["B"+dir +"_wd"]
            + 2 * b,
            hatch="/",
            color=c0["wd"],
        )
        ax.barh(
            1 * a,
            -df["E"+dir +"_ggd"],
            left=-b + df["E"+dir +"_ggd"],
            color=c0["ggd"],
            hatch="/",
        )

    ax.text(
        ts / 2,
        1 * a + 0.5,
        r"Balanced and errors parts MS $\beta_i$ and $\epsilon_i$",
        ha="center",
    )

    # percentage + MS
    key = ["B"+dir +"_acc", "E"+dir +"_acc", "B"+dir +"_cor", "E"+dir +"_cor", "B"+dir +"_ggd", "E"+dir +"_ggd", "B"+dir +"_wd", "E"+dir +"_wd"]
    for i in range(len(key)):
        d = 0  # vertical +
        dx = 0  # horizontal + on MS
        dxx = 0  # horizontal + on percentage
        if i == len(key) - 1:
            d = -0.1 * a
            dx = 3e-11 / U2
            dxx = 1.5e-11 / U2
        if i == len(key) - 2:
            d = 0.1 * a
        if (
            abs(int(np.rint((df[key[i]] / ts) * 100))) > 0
        ):  # does not plot percentage below 1%
            if df[key[i]] > 0:
                ax.text(
                    sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b / 2 + dxx,
                    a + d,
                    f"{int(np.rint((df[key[i]]/ts)*100))} %",
                    ha="center",
                    bbox=bbox,
                )
            else:
                ax.text(
                    df[key[i]] / 2 - dxx,
                    a + d,
                    f"{int(np.rint((df[key[i]]/ts)*100))} %",
                    ha="center",
                    bbox=bbox,
                )
        d = 0
        if i % 2 == 1:
            d = -0.1 * a
        ax.text(
            sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b / 2 + dx,
            a + d - 0.55,
            f"{np.round(df[key[i]],2)}",
            ha="center",
        )

    ## PAIRS + RESIDUAL ##
    plt.rcParams["hatch.linewidth"] = 8
    plt.rcParams["hatch.color"] = c0["ggd"]
    ax.barh(0, df["X"+dir +"_cor_ggd"], color=c0["cor"], hatch="/")
    plt.rcParams["hatch.color"] = c0["cor"]
    ax.barh(0, df["X"+dir +"_acc_cor"], color=c0["acc"], hatch="/", left=df["X"+dir +"_cor_ggd"] + b)
    plt.rcParams["hatch.color"] = c0["acc"]
    ax.barh(
        0,
        df["X"+dir +"_acc_ggd"],
        color=c0["ggd"],
        hatch="/",
        left=df["X"+dir +"_cor_ggd"] + df["X"+dir +"_acc_cor"] + 2 * b,
    )
    plt.rcParams["hatch.color"] = c0["wd"]
    ax.barh(
        0,
        df["X"+dir +"_cor_wd"],
        color=c0["cor"],
        hatch="/",
        left=df["X"+dir +"_cor_ggd"] + df["X"+dir +"_acc_cor"] + df["X"+dir +"_acc_ggd"] + 3 * b,
    )
    ax.barh(
        0,
        df["S"+dir],
        label="Errors",
        color="lightgrey",
        left=df["X"+dir +"_cor_ggd"]
        + df["X"+dir +"_acc_cor"]
        + df["X"+dir +"_acc_ggd"]
        + df["X"+dir +"_cor_wd"]
        + 4 * b,
    )
    # negative contribution
    plt.rcParams["hatch.color"] = c0["acc"]
    ax.barh(0, -df["X"+dir +"_acc_wd"], color=c0["wd"], hatch="/", left=df["X"+dir +"_acc_wd"] - b)
    plt.rcParams["hatch.color"] = c0["ggd"]
    ax.barh(
        0,
        -df["X"+dir +"_ggd_wd"],
        color=c0["wd"],
        hatch="/",
        left=df["X"+dir +"_acc_wd"] + df["X"+dir +"_ggd_wd"] - 2 * b,
    )

    tts = (
        df["X"+dir +"_cor_ggd"]
        + df["X"+dir +"_acc_cor"]
        + df["X"+dir +"_acc_ggd"]
        + df["X"+dir +"_cor_wd"]
        + 4 * b
        + df["S"+dir]
    )
    print(tts)
    sum_pairs = (
        df["X"+dir +"_cor_ggd"] + df["X"+dir +"_acc_cor"] + df["X"+dir +"_acc_ggd"] + df["X"+dir +"_cor_wd"] + 3 * b
    )
    ax.text(sum_pairs / 2, 0.6, r"Pairs' contributions $X_{ij}$", ha="center")

    # accolade
    c = 1e-12
    id1 = 0
    id2 = sum_pairs
    bx = [id1, id1, id2, id2]
    by = [0.45, 0.5, 0.5, 0.45]
    # ax.plot(bx, by, 'k-', lw=2)
    ax.text(sum_pairs + df["S"+dir] / 2, 0.5, r"$S$", ha="center")

    # percentage + MS
    from itertools import combinations

    correlation = list(combinations(["acc", "cor", "GGD", "wd"], 2))
    key = ["X"+dir +"_cor_ggd", "X"+dir +"_acc_cor", "X"+dir +"_acc_ggd", "X"+dir +"_cor_wd"]
    for i in range(len(key)):
        d = 0
        if aviso and key[i] == "X"+dir +"_acc_ggd":
            d = -0.1 * a

        if (
            abs(int(np.rint((df[key[i]] / ts) * 100))) > 0
        ):  # does not plot percentage below 1%
            ax.text(
                sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b,
                0 + d * 2,
                f"{int(np.rint((df[key[i]]/ts)*100))} %",
                ha="center",
                bbox=bbox,
            )

        ax.text(
            sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b,
            0 - 0.55 + d,
            f"{np.round(df[key[i]],2)}",
            ha="center",
        )

    # negative contribution
    key = ["X"+dir +"_acc_wd", "X"+dir +"_ggd_wd"]
    for i in range(len(key)):
        if abs(int(np.rint((df[key[i]] / ts) * 100))) > 0:
            ax.text(
                sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b,
                0,
                f"{int(np.rint((df[key[i]]/ts)*100))} %",
                ha="center",
                bbox=bbox,
            )
        d = 0
        if i % 2 == 1:
            d = -0.1 * a
        ax.text(
            sum([df[v] for v in key[:i]]) + df[key[i]] / 2 + i * b,
            0 - 0.55 + d,
            f"{np.round(df[key[i]],2)}",
            ha="center",
        )

    key = ["X"+dir +"_cor_ggd", "X"+dir +"_acc_cor", "X"+dir +"_acc_ggd", "X"+dir +"_cor_wd"]
    ax.text(
        sum([df[v] for v in key]) + df["S"+dir] / 2 + i * b,
        0,
        f'{int(np.rint((df["S"+dir]/ts)*100))} %',
        ha="center",
        bbox=bbox,
    )
    ax.text(
        sum([df[v] for v in key]) + df["S"+dir] / 2 + i * b,
        0 - 0.55,
        f'{np.round(df["S"+dir],2)}',
        ha="center",
    )

    # FIGURE SET
    ax.set_yticks([])
    if not xlim:
        xlim = tts
    ax.axvline(0, ls=":", c="grey")
    ax.set_xlim(-0.5, xlim + 0.5)
    ax.set_ylim(-1, 4.1)
    ax.get_yaxis().set_visible(False)
    ax.annotate(
        "",
        xy=(xlim, -1),
        xytext=(-0.5, -1),
        arrowprops={"arrowstyle": "->", "facecolor": "k"},
    )
    ax.set_xlabel(r"$[\gamma^2]$")