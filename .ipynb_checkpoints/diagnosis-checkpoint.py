import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import os
from glob import glob

from cstes import c0, U2, zarr_dir, surface_drifters, depth_drifters, depth_100, depth_50


""" CREATE DATASET """

drifters_sources = 'all_med_variational_10min_v0.nc'
#drifters_sources = 'all_med_lowess_10min_v0.nc'

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
    
def create_id_comb(drifter_key, alti_key, wd_key = 'era5', ggd_var='etaf', wd_depth='0', wd_model='rio'):
    return drifter_key + '__' + alti_key +'_' + ggd_var + '__' + wd_model+'_'+ wd_depth + wd_key

def select_nearest_swot_coloc(df):
    idx = df.groupby(['pass_number','cycle_number','drifter_id']).time_to_swot.idxmin()
    return df.loc[idx]

def one_coloc(drifter_key, alti_key, wd_key = 'era5', ggd_var='etaf', wd_depth='0', wd_model='rio',):
    df = pd.concat([prepared_drifters(drifter_key),
                    prepared_alti(alti_key)[['ggde_'+ggd_var,'ggdn_'+ggd_var]].rename(columns = {'ggde_'+ggd_var:'ggde','ggdn_'+ggd_var:'ggdn'}), 
                    prepared_wd(wd_key)[['wde'+wd_depth+'_'+wd_model,'wdn'+wd_depth+'_'+wd_model]].rename(columns = {'wde'+wd_depth+'_'+wd_model : 'wde','wdn'+wd_depth+'_'+wd_model:'wdn'}),
                   ], axis=1).dropna() # with dropna, depends on the altimetry filter
    id_comb = create_id_comb(drifter_key, alti_key, wd_key, ggd_var, wd_depth, wd_model)
    
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
    var = ['acc', 'cor', 'ggd', 'wd']
    eneg = [ v for v in var if df["E"+dir +"_"+v]<0]
    epos = [ v for v in var if df["E"+dir +"_"+v]>=0]
    spos, sneg = 0, 0
    ipos, ineg = 0, 1
    
    #text parameters
    d=0# vertical +
    db = 0.15  # vertical + 
    de = -0.2  # vertical +
    dx = 0  # horizontal + on MS
    dxx = 0  # horizontal + on percentage
    for v in var :
        ax.barh(1 * a, df["B"+dir +"_"+v], left=spos+ipos*b, color=c0[v])
        if (abs(int(np.rint((df["B"+dir +"_"+v] / ts) * 100))) > 0):# does not plot percentage below 1%
            ax.text(
                spos + df["B"+dir +"_"+v]/ 2 + ipos * b / 2 + dxx,
                    a + db,
                    f'{int(np.rint((df["B"+dir +"_"+v]/ts)*100))} %',
                    ha="center",
                    bbox=bbox,
                )
            ax.text(
                spos + df["B"+dir +"_"+v] / 2 + i * b / 2 + dx,
                a + d - 0.55,
                f'{np.round(df["B"+dir +"_"+v],2)}',
                ha="center",
                )
        spos += df["B"+dir +"_"+v]
            
        if v in epos : 
            ax.barh(1 * a, df["E"+dir +"_"+v], left=spos + ipos*b, color=c0[v], hatch="/")
            ax.text(
                spos + df["E"+dir +"_"+v]/ 2 + ipos * b / 2 + dxx,
                a + de,
                f'{int(np.rint((df["E"+dir +"_"+v]/ts)*100))} %',
                ha="center",
                bbox=bbox,
                )
            ax.text(
                spos + df["E"+dir +"_"+v] / 2 + i * b / 2 + dx,
                a + d - 0.7,
                f'{np.round(df["E"+dir +"_"+v],2)}',
                ha="center",
                )
            spos += df["E"+dir +"_"+v]
            ipos += 1
            
        
        if v in eneg : 
            ax.barh(1 * a,-df["E"+dir +"_"+v], left=-b -b*ineg + df["E"+dir +"_"+v]+sneg,color=c0[v],hatch="/",)
            if (abs(int(np.rint((df["B"+dir +"_"+v] / ts) * 100))) > 0):
                ax.text(
                    sneg + df["E"+dir +"_"+v]/ 2 + ineg * b / 2 + dxx,
                    a + d,
                    f'{int(np.rint((df["E"+dir +"_"+v]/ts)*100))} %',
                    ha="center",
                    bbox=bbox,
                    )
                ax.text(
                    sneg + df["E"+dir +"_"+v] / 2 + i * b / 2 + dx,
                    a + d - 0.55,
                    f'{np.round(df["E"+dir +"_"+v],2)}',
                    ha="center",
                )
            sneg += df["E"+dir +"_"+v]
            ineg +=1

    ax.text(ts / 2,1 * a + 0.5, r"Balanced signal and residual contributions $\beta_i$ and $\mathcal{E}_i$", ha="center",)


    ## PAIRS + RESIDUAL ##
    comb = [("cor", "ggd"), ("acc", "cor"), ("acc", "ggd"), ("cor", "wd"), ("ggd", "wd"), ("acc", "wd")]
    plt.rcParams["hatch.linewidth"] = 8
    xpos = [c for c in comb if df["X"+dir +"_"+c[0]+"_"+c[1]]>=0]
    spos, sneg = 0, 0
    ipos, ineg = 0, 1

    for c in comb :
        plt.rcParams["hatch.color"] = c0[c[1]]
        #positive pairs contributions
        if c in xpos :
            ax.barh(0, df["X"+dir +"_"+c[0]+"_"+c[1]], color=c0[c[0]], hatch="/", left=spos + b*ipos)
            if (abs(int(np.rint((df["X"+dir +"_"+c[0]+"_"+c[1]] / ts) * 100))) > 0):  # does not plot percentage below 1%
                ax.text(
                    spos + df["X"+dir +"_"+c[0]+"_"+c[1]] / 2 + i * b,
                    0 + d * 2,
                    f'{int(np.rint((df["X"+dir +"_"+c[0]+"_"+c[1]]/ts)*100))} %',
                    ha="center",
                    bbox=bbox,
                )
    
                ax.text(
                    spos + df["X"+dir +"_"+c[0]+"_"+c[1]] / 2 + i * b,
                    0 - 0.55 + d,
                    f'{np.round(df["X"+dir +"_"+c[0]+"_"+c[1]],2)}',
                    ha="center",
                )
            spos += df["X"+dir +"_"+c[0]+"_"+c[1]]
            ipos += 1

        
        # negative contributions
        else : 
            dp, dj = 0.19, 0
            if ineg%2 ==0 : 
                dp = -dp
                dj = -0.17

            ax.barh(0, -df["X"+dir +"_"+c[0]+"_"+c[1]], color=c0[c[0]], hatch="/", left=df["X"+dir +"_"+c[0]+"_"+c[1]]+sneg - b*ineg)
            if (abs(int(np.rint((df["X"+dir +"_"+c[0]+"_"+c[1]] / ts) * 100))) > 0):  # does not plot percentage below 1%
                ax.text(
                    sneg + df["X"+dir +"_"+c[0]+"_"+c[1]] / 2 + i * b,
                    0 + d * 2+dp,
                    f'{int(np.rint((df["X"+dir +"_"+c[0]+"_"+c[1]]/ts)*100))} %',
                    ha="center",
                    bbox=bbox,
                )
    
                ax.text(
                    sneg + df["X"+dir +"_"+c[0]+"_"+c[1]] / 2 + i * b,
                    0 - 0.55 + dj,
                    f'{np.round(df["X"+dir +"_"+c[0]+"_"+c[1]],2)}',
                    ha="center",
                )
            sneg += df["X"+dir +"_"+c[0]+"_"+c[1]]
            ineg += 1
    
    # Residual         
    ax.barh(0,df["S"+dir],label="Errors",color="lightgrey",left=spos+ ipos * b,)
    ax.text(
            spos + df["S"+dir] / 2 + i * b,
            0 - 0.55 + d,
            f'{np.round(df["S"+dir],2)}',
            ha="center",
        )
    ax.text(
        spos + df["S"+dir] / 2 + i * b,
        0 + d * 2,
        f'{int(np.rint((df["S"+dir]/ts)*100))} %',
        ha="center",
        bbox=bbox,
    )
    

    tts = (spos
        + 4 * b
        + df["S"+dir]
    )
    print(tts)
    ax.text(spos / 2, 0.6, r"Pairs' contributions $X_{ij}$", ha="center")

    # accolade
    c = 1e-12
    id1 = 0
    id2 = ts
    bx = [id1, id1, id2, id2]
    by = [0.45, 0.5, 0.5, 0.45]
    # ax.plot(bx, by, 'k-', lw=2)
    ax.text(spos + df["S"+dir] / 2, 0.5, r"$\mathcal{E}$", ha="center")


    # FIGURE SET
    ax.set_yticks([])
    if not xlim:
        xlim = (sneg, spos + df["S"+dir]+0.3)
    ax.axvline(0, ls=":", c="grey")
    ax.set_xlim(xlim[0], xlim[1] + 0.5)
    ax.set_ylim(-1, 4.1)
    ax.get_yaxis().set_visible(False)
    ax.annotate(
        "",
        xy=(xlim[1], -1),
        xytext=(-0.5, -1),
        arrowprops={"arrowstyle": "->", "facecolor": "k"},
    )
    ax.set_xlabel(r"$[\gamma^2]$")



def compute_mean_square(ds):
    var = ['acc', 'cor', 'ggd', 'wd','sum']
    VAR = ['ACC', 'COR', 'GGD', 'WD', 'S']
    dss = (ds[[v+'n' for v in var] + [v+'e' for v in var]]**2).mean(dim='row_number').rename({var[i]+'n' : VAR[i]+'n' for i in range(5)}).rename({var[i]+'e' : VAR[i]+'e' for i in range(5)})
    dss['sigmae'] = dss.ACCe + dss.CORe + dss.GGDe + dss.WDe
    dss['sigman'] = dss.ACCn + dss.CORn + dss.GGDn + dss.WDn
    dss['nb_coloc'] = len(ds.row_number)
    #Balanced and error
    for direction in ['e', 'n']:
        for v in ['acc', 'cor', 'ggd', 'wd'] : 
                dss['B'+direction+'_'+v] = -((ds[v+direction]*ds['exc'+direction+'_'+v])).mean(dim='row_number')
                dss['E'+direction+'_'+v] = ((ds[v+direction]*ds['sum'+direction])).mean(dim='row_number')
    #pairs contributions
    for v in [v for v in ds if 'prod' in v]:
        dss[v.replace('prod', 'X')]=-2*ds[v].mean(dim='row_number')
    return dss