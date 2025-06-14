import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import os
from glob import glob

from cstes import c0, U2, zarr_dir, surface_drifters, depth_drifters, depth_100, depth_50, images_dir
from swot import browse_swot_250m, browse_swot_2km

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
crs = ccrs.PlateCarree()

import cartopy.geodesic as geod
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pyproj
from pyproj import Geod

from cstes import drifters_sources

import matplotlib.transforms as mtransforms
def put_fig_letter(fig, ax, letter):
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        letter + ")",
        transform=ax.transAxes + trans,
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0),
        zorder=30,
    )
    
""" 
_________________________________________
---- CREATE DATASETS ----
_________________________________________
"""
def define_coloc_source(dt, drifter_preprocess = '', drifter_preprocess_param='') :
    #default 
    file_key, spectral_key='spectral_decomp', '' #spectral_decomp files contain original velocities/accelerations
    
    # assert
    assert drifter_preprocess in ['', 'spectral_decomp', 'low_pass'], "drifter_preprocess should be in ['', 'spectral_decomp', 'low_pass']"
    
    # spectral decomp
    if drifter_preprocess == 'spectral_decomp' : 
        assert  drifter_preprocess_param in ['', 'LF', 'inertial', 'diurnal', 'semidiurnal', 'HF'], "drifter_preprocess_param while using spectral_decomp should be in  ['', 'LF', 'inertial', 'diurnal', 'semidiurnal', 'HF']"
        spectral_key = drifter_preprocess_param+'_'
        file_key = 'spectral_decomp'
        
    # low pass    
    if drifter_preprocess == 'low_pass' : 
        cutoff = drifter_preprocess_param
        file_key = f"low_pass_{str(cutoff).replace('0.', '0')}"

    # find good file
    return f'{dt}_{file_key}_'+drifters_sources.replace('.nc', '')
    

dtypes = {'drifter_id':str, 'drifter_type':str}

def prepared_drifters(dt, drifter_preprocess = '', drifter_preprocess_param='') :
    
    colocs_sources = define_coloc_source(dt, drifter_preprocess, drifter_preprocess_param)
    drifters_path = os.path.join(zarr_dir, 'coloc_files', 'drifters', f'drifterscoloc_'+colocs_sources+'.csv')

    if (drifter_preprocess == 'spectral_decomp') & (drifter_preprocess_param !='') : spectral_key = drifter_preprocess_param+'_'
    else : spectral_key=''
            
    # Treat
    dfr = pd.read_csv(drifters_path, parse_dates=['datetime'], dtype=dtypes).set_index('row_number')
    dfr['time_to_swot'] = dfr.time_to_swot.astype("timedelta64[ns]")
    dfr['f'] =  2 * 2 * np.pi / 86164.1 * np.sin(dfr.latitude * np.pi / 180)
    dfr['core']= -dfr.f * dfr[spectral_key + 'velocity_north']
    dfr['corn']= dfr.f * dfr[spectral_key + 'velocity_east']
    dfr = dfr.rename(columns = {spectral_key + 'acceleration_north':'accn', spectral_key + 'acceleration_east':'acce'})
    return dfr[['datetime', 'longitude', 'latitude', 'pass_number','time_to_swot','cycle_number','cycle_date','drifter_id','drifter_type','accn', 'acce', 'core', 'corn']]


def prepared_alti(dt, drifter_preprocess = '', drifter_preprocess_param='', alti_product_key='swot2km', alti_diff_method = 'diff_only', alti_diff_method_param = '') :
    
    colocs_sources = define_coloc_source(dt, drifter_preprocess, drifter_preprocess_param)

    # L4 products
    if 'L4' in alti_product_key :
        dfs = pd.read_csv(os.path.join(zarr_dir, "coloc_files",'alti', 'alticoloc_'+alti_product_key+'_'+colocs_sources+'.csv')).set_index('row_number')
        dfs['f'] =  2 * 2 * np.pi / 86164.1 * np.sin(dfs.latitude * np.pi / 180)
        dfs['ggde_fromduacsv'] = dfs.f * dfs.vgos
        dfs['ggdn_fromduacsv'] = -dfs.f *dfs.ugos
        l = ['ggde_fromduacsv','ggdn_fromduacsv'] 
        dfs = dfs[l]
    
    #L3 products
    else : 
        #General
        dfg = pd.read_csv(os.path.join(zarr_dir, 'coloc_files', 'alti',f'alticoloc_{alti_product_key}_general_'+colocs_sources+'.csv')).set_index('row_number')
        dfg['f'] =  2 * 2 * np.pi / 86164.1 * np.sin(dfg.latitude * np.pi / 180)
        
        if alti_diff_method !='fromduacsv':
            # SSH
            if alti_diff_method_param != '': alti_diff_method_param = str(alti_diff_method_param)+'_'
            dfs = pd.read_csv(os.path.join(zarr_dir, 'coloc_files', 'alti',f'alticoloc_{alti_product_key}_{alti_diff_method}_{alti_diff_method_param}'+colocs_sources+'.csv')).set_index('row_number')
            dfg = pd.concat([dfg, dfs], axis=1)
        
        # from duacs
        dfs = dfg.copy()#defragmented
        dfs['ggde_fromduacsv'] = dfs.f * dfs.duacs_speed_meridional_abs
        dfs['ggdn_fromduacsv'] = -dfs.f *dfs.duacs_speed_zonal_abs

        l =  ['ggde_fromduacsv', 'ggdn_fromduacsv', 'phi', 'distance_to_coast']
        
        if alti_diff_method !='fromduacsv':
            ggd_var = [v.replace('dx_', '') for v in dfs.columns if (('etaf' in v)|('etac' in v))&('dx_' in v)]
            # rotate and compute quantities
            for v in ggd_var :
                g=9.81
                from swot import rotate
                dfs['ggde_'+v] = g * rotate(dfs['dx_'+v], dfs['dy_'+v], dfs['phi'])[0]
                dfs['ggdn_'+v] = g * rotate(dfs['dx_'+v], dfs['dy_'+v], dfs['phi'])[1]
                dfs['vorticity_'+v] = g * (dfs['dxx_'+v] + dfs['dyy_'+v])/dfs.f**2
                dfs['strain_s_'+v] = g * (dfs['dxx_'+v] - dfs['dyy_'+v])/dfs.f**2
                dfs['strain_n_'+v] = -2 * g *(dfs['dxy_'+v] - dfs['dxy_'+v])/dfs.f**2

            l = l+['ggde_'+v for v in ggd_var]+['ggdn_'+v for v in ggd_var]+['vorticity_'+v for v in ggd_var]+['strain_s_'+v for v in ggd_var]+['strain_n_'+v for v in ggd_var]

        dfs = dfs[l]
        
    return dfs.sort_index()#.rename(columns ={v : v+'_' +alti_product_key for v in dfs})


def prepared_wd(dt, drifter_preprocess = '', drifter_preprocess_param='', wd_product_key = 'era5', wd_model ='rio'):
    
    colocs_source = define_coloc_source(dt, drifter_preprocess, drifter_preprocess_param)
    if wd_model in 'rioagesc': wd_model = 'rioagesc' #both in one file
    if os.path.isfile(os.path.join(zarr_dir,'coloc_files','wind',f'{wd_product_key}_{wd_model}_'+colocs_source.replace('.nc', '')+'.csv')):
        dfw = pd.read_csv(os.path.join(zarr_dir,'coloc_files','wind',f'{wd_product_key}_{wd_model}_'+colocs_source.replace('.nc', '')+'.csv')).set_index('row_number')
    if os.path.isfile(os.path.join(zarr_dir,'coloc_files','wind',f'{wd_product_key}_{wd_model}_'+colocs_source.replace('.nc', '')+'.parquet')):
        dfw = pd.read_parquet(os.path.join(zarr_dir,'coloc_files','wind',f'{wd_product_key}_{wd_model}_'+colocs_source.replace('.nc', '')+'.parquet'))#.set_index('row_number')
    dfw = dfw[[v for v in dfw if ('vsde' in v or 'vsdn' in v)]]
    return dfw#.rename(columns ={v : v+'_' +wd_key for v in dfw})

from cstes import surface_drifters, depth_drifters, depth_50, depth_100

def create_id_comb(dt, 
                   drifter_preprocess,
            drifter_preprocess_param,
            alti_product_key,
            alti_diff_method,
            alti_diff_method_param,
            ggd_var,
            wd_product_key, 
            wd_model, 
            wd_depth):
    
    #For coherence
    if 'L4' in alti_product_key : 
        alti_diff_method = 'fromduacsv'
        alti_diff_method_param =''
        ggd_var = 'fromduacsv'

    else : l = ['ggde_'+ggd_var,'ggdn_'+ggd_var, 'phi', 'distance_to_coast']

    if ggd_var == 'fromduacsv':
        alti_diff_method = 'fromduacsv'
        alti_diff_method_param =''
        
    colocs_source = define_coloc_source(dt, drifter_preprocess, drifter_preprocess_param)
    return f"{colocs_source}__{drifter_preprocess_param}__{alti_product_key}_{alti_diff_method}_{alti_diff_method_param}_{ggd_var}__{wd_product_key}_{wd_model}{wd_depth}"

def select_nearest_swot_coloc(df):
    idx = df.groupby(['pass_number','cycle_number','drifter_id']).time_to_swot.idxmin()
    return df.loc[idx]

def one_comb(dt, 
            drifter_preprocess,
            drifter_preprocess_param,
            alti_product_key,
            alti_diff_method,
            alti_diff_method_param,
            ggd_var,
            wd_product_key, 
            wd_model, 
            wd_depth):
    
    id_comb = create_id_comb(dt, drifter_preprocess, drifter_preprocess_param, alti_product_key, alti_diff_method, alti_diff_method_param, ggd_var, wd_product_key, wd_model, wd_depth)

    #For coherence
    if 'L4' in alti_product_key : 
        alti_diff_method = 'fromduacsv'
        alti_diff_method_param =''
        ggd_var = 'fromduacsv'
        print("only 'fromduacsv' method is available for L4")
        l = ['ggde_'+ggd_var,'ggdn_'+ggd_var]

    else : l = ['ggde_'+ggd_var,'ggdn_'+ggd_var, 'phi', 'distance_to_coast']

    if (ggd_var == 'fromduacsv') | (alti_diff_method =='fromduacsv'):
        ggd_var = 'fromduacsv'
        alti_diff_method = 'fromduacsv'
        alti_diff_method_param =''

    
    df = pd.concat([prepared_drifters(dt, drifter_preprocess, drifter_preprocess_param),
                    prepared_alti(dt, drifter_preprocess, drifter_preprocess_param, alti_product_key, alti_diff_method, alti_diff_method_param)[l].rename(columns = {'ggde_'+ggd_var:'ggde','ggdn_'+ggd_var:'ggdn'}), 
                    prepared_wd(dt, drifter_preprocess, drifter_preprocess_param, wd_product_key, wd_model)[['vsde_'+wd_model + '_z'+wd_depth,'vsdn_'+wd_model + '_z'+wd_depth]].rename(columns = {'vsde_'+wd_model + '_z'+wd_depth: 'wde','vsdn_'+wd_model + '_z'+wd_depth:'wdn'}),
                   ], axis=1).dropna() # with dropna, depends on the altimetry filter
    print(len(df))
    
    if 'phi' not in df.columns :
        df['phi'] = np.zeros(len(df))
    if 'distance_to_coast' not in df.columns :
        df['distance_to_coast'] = np.zeros(len(df))
    
    #depth
    depth = df['pass_number'].copy()
    depth.loc[df['drifter_type'].isin(surface_drifters)]=0
    depth.loc[df['drifter_type'].isin(depth_drifters)]=15
    depth.loc[df['drifter_id'].isin(depth_50)]=50
    depth.loc[df['drifter_id'].isin(depth_100)]=100

    df['depth'] = depth
    print(id_comb)
    
    return df, id_comb

def create_prod_exc_sum(df,id_comb, null_mean = False):
    coords_list = ['datetime', 'longitude', 'latitude', 'pass_number','time_to_swot','cycle_number','drifter_id', 'drifter_type','depth', 'phi', 'distance_to_coast']
    ds = df[coords_list].to_xarray()
    ds['id_comb'] = id_comb

    for direction in ['e', 'n'] : 
            if direction == 'e' : l = {'acc':'acce', 'cor':'core', 'ggd':'ggde', 'wd':'wde'}
            if direction == 'n' : l = {'acc':'accn', 'cor':'corn', 'ggd':'ggdn', 'wd':'wdn'}

            if null_mean : 
                for v in l:
                    df[l[v]] = df[l[v]] - df[l[v]].mean()
                
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
    return ds


def dataset_coloc_combs(comb_list, nearest =False, remove_id_outliers = True, null_mean = False):
    coords_list = ['datetime', 'longitude', 'latitude', 'pass_number','time_to_swot','cycle_number','drifter_id', 'drifter_type','depth', 'phi', 'distance_to_coast']
    D = []
    for comb in comb_list :
        df, id_comb = one_comb(**comb)

        # remove identified swot outliers
        if remove_id_outliers :
            from cstes import remove_pb_swot
            df = remove_pb_swot(df)
            
        if nearest :
            # select only nearest in time colocalisation
            df = select_nearest_swot_coloc(df)#.dropna()
        
        ds = create_prod_exc_sum(df, id_comb, null_mean)
        
        D.append(ds.set_coords('id_comb'))
    ds = xr.concat(D, dim='id_comb').set_coords(coords_list)
        
    # remove identified swot outliers
    if remove_id_outliers :
        from cstes import err_acc
        err_acc_update = np.array(err_acc)[np.isin(err_acc, ds.row_number)]
        ds = ds.drop_sel(row_number = err_acc_update)  
        print('Identified SWOT and drifter outliers have been removed')
        
    return ds

def update_dict(base_dic, kwargs):
    dic= base_dic.copy()
    dic.update(**kwargs)
    return dic

def extract_mean_geo(DS, dirname = ('e', 'n')):
    DSt = DS.copy()
    for dir_ in dirname:
        ggdm = DS['ggd'+dir_].mean('row_number').values
        corm = DS['cor'+dir_].mean('row_number').values
        delta = xr.DataArray(data = np.where(abs(ggdm)>abs(corm), abs(corm), abs(ggdm)), coords = dict(id_comb = DS.id_comb))
    
        DSt['meancor'+dir_] = np.sign(corm)*delta
        DSt['meanggd'+dir_] = np.sign(ggdm)*delta
        
        DSt['ggd'+dir_] = DS['ggd'+dir_]-np.sign(ggdm)*delta
        DSt['cor'+dir_] = DS['cor'+dir_]-np.sign(corm)*delta
        
        l = {'acc':'acc'+dir_, 'cor':'cor'+dir_, 'ggd':'ggd'+dir_, 'wd':'wd'+dir_}
        # sum 
        DSt['sum'+dir_] = sum([DSt[l[v]] for v in l])
        
        l = {'acc':'acc'+dir_, 'cor':'cor'+dir_, 'ggd':'ggd'+dir_, 'wd':'wd'+dir_}
        # simple var + sum- one term
        for v in l :
            DSt['exc'+dir_+'_'+v] = DSt['sum'+dir_]-DSt[l[v]]
            
        #2 by 2 product  
        l = {'acc':'acc'+dir_, 'cor':'cor'+dir_, 'ggd':'ggd'+dir_, 'wd':'wd'+dir_}
        import itertools  
        couple = list(itertools.combinations(list(l.keys()),2))
        for c in couple : 
            DSt['prod'+dir_+'_'+'_'.join(c)] = DSt[c[0]+dir_] * DSt[c[1]+dir_]
            
    return DSt

""" 
_________________________________________
---- CLOSURE ANALYSIS FUNCTIONS ----
_________________________________________
"""


closure_vars = ['ACC*', 'COR*', 'GGD*', 'WD*', 'S*', 'sigma*'] + ['B*_'+v for v in ['acc', 'cor', 'ggd', 'wd']] + ['E*_'+v for v in ['acc', 'cor', 'ggd', 'wd']] + ['X*_acc_cor', 'X*_acc_ggd', 'X*_acc_wd', 'X*_cor_ggd', 'X*_cor_wd', 'X*_ggd_wd'] + ['D*_cyclo', 'D*_anticyclo']

def compute_mean_square(ds, dirname = ('e', 'n')):
    """ Compute closure stats
    ds : dataset containing terms values for all row_numbers
    dirname : directions of the reconstruction
    """
    d0, d1 = dirname[0], dirname[1]

    closure_vars_=closure_vars
    
    var = ['acc', 'cor', 'ggd', 'wd','sum']
    VAR = ['ACC', 'COR', 'GGD', 'WD', 'S']
    if 'meancor'+dirname[0] in ds :
        print('ok')
        var += ['meancor', 'meanggd']
        VAR += ['meancor'.upper(), 'meanggd'.upper()]
        closure_vars_ = closure_vars +['MEANCOR*', 'MEANGGD*']
        
    dss = (ds[[v+d1 for v in var] + [v+d0 for v in var]]**2).rename({var[i]+d1 : VAR[i]+d1 for i in range(len(var))}).rename({var[i]+d0 : VAR[i]+d0 for i in range(len(var))})
    dss['sigma'+d0] = dss['ACC'+d0] + dss['COR'+d0] + dss['GGD'+d0] + dss['WD'+d0]
    dss['sigma'+d1] = dss['ACC'+d1] + dss['COR'+d1] + dss['GGD'+d1] + dss['WD'+d1]
    
    nb_coloc = len(ds.row_number)
    
    #Balanced and error
    for direction in [d0, d1]:
        for v in ['acc', 'cor', 'ggd', 'wd'] : 
            dss['B'+direction+'_'+v] = -((ds[v+direction]*ds['exc'+direction+'_'+v]))
            dss['E'+direction+'_'+v] = ((ds[v+direction]*ds['sum'+direction]))
            
    #pairs contributions
    for v in [v for v in ds if 'prod' in v]:
        dss[v.replace('prod', 'X')]=-2*ds[v]

    # Cyclo/anticyclo contribution
    for direction in [d0, d1]:
        dss['D'+direction+'_cyclo'] = -2 *(ds['acc'+direction]+ds['cor'+direction])*ds['ggd'+direction]
        dss['D'+direction+'_anticyclo'] = -2 *(ds['acc'+direction]+ds['ggd'+direction])*ds['cor'+direction]

    #Sum of both direction
    for v in closure_vars_ :
        dss[v.replace('*', '')] = dss[v.replace('*', dirname[0])] + dss[v.replace('*', dirname[1])]

    # Mean
    dsm = dss.mean('row_number')
    
    # Add errors central limit
    dsse = (2*dss.std('row_number')/np.sqrt(nb_coloc)).rename({v:'ser__'+v for v in list(dss.keys())})

    dss = xr.merge([dsm, dsse])

    #Mean geostrophics corrections :
    if 'meancor'+dirname[0] in ds :
        print('geostrophic mean corrections')
        for dir_ in dirname:
            for v in ['cor', 'ggd']:
                dss[v.upper() + dir_] = dss[v.upper() + dir_] + dss['MEAN'+ v.upper() + dir_]
                dss['B'+ dir_+'_'+v] = dss['B'+ dir_+'_'+v] + dss['MEAN'+ v.upper() + dir_]
        dss['X'+ dir_+'_cor_ggd'] = dss['X'+ dir_+'_cor_ggd'] + 2*dss['MEAN'+ v.upper() + dir_]
        for v in ['cor', 'ggd']:
            dss[v.upper()] =  dss[v.upper() + dirname[0]]+ dss[v.upper() + dirname[1]]
            dss['B_'+v] = dss['B'+dirname[0]+'_'+v] + dss['B'+dirname[1]+'_'+v]
            dss['X_cor_ggd'] = dss['X'+dirname[0]+'_cor_ggd'] + dss['X'+dirname[1]+'_cor_ggd']
        
    #end
    dss = assign_attrs(dss, list(dirname) + [''])
    return dss


def compute_variance(ds_, dirname = ('e', 'n')):
    """ Compute closure stats
    ds : dataset containing terms values for all row_numbers
    dirname : directions of the reconstruction
    """
    
    var = ['acc', 'cor', 'ggd', 'wd','sum']
    VAR = ['ACC', 'COR', 'GGD', 'WD', 'S']

    ds=ds_.copy()
    # remove mean
    for dir_ in dirname : 
        for v in var:
            ds[v+dir_] = ds[v+dir_]-ds[v+dir_].mean('row_number')
            
        # sum 
        ds['sum'+dir_] = sum([ds[v+dir_] for v in var])

        # simple var + sum- one term
        for v in var :
            ds['exc'+dir_+'_'+v] = ds['sum'+dir_]-ds[v+dir_]
                
        #2 by 2 product    
        import itertools
        l = {'acc':'acc'+dir_, 'cor':'cor'+dir_, 'ggd':'ggd'+dir_, 'wd':'wd'+dir_}
        couple = list(itertools.combinations(list(l.keys()),2))
        for c in couple : 
            ds['prod'+dir_+'_'+'_'.join(c)] = ds[c[0]+dir_] * ds[c[1]+dir_]

    # MS = var as mean are null
    dss = compute_mean_square(ds, dirname)
    return dss


def assign_attrs(ds, dirname=('e', 'n', '')) :
    """ Assign attributes to compute_mean_square() functions dataset output """
    for dir_ in dirname :
        if dir_ == 'e': Dir = 'Zonal'
        if dir_ == 'n': Dir = 'Meridional'
        if dir_ == 'x': Dir = 'Cross-track'
        if dir_ == 'y': Dir = 'Along-track'
        if dir_ == '': Dir = 'Total'

        
        ds['ACC'+dir_] = ds['ACC'+dir_].assign_attrs({'long_name':Dir + ' Lagrangian acceleration MS'})
        ds['COR'+dir_] = ds['COR'+dir_].assign_attrs({'long_name':Dir + ' Coriolis acceleration MS'})
        ds['GGD'+dir_] = ds['GGD'+dir_].assign_attrs({'long_name':Dir + ' Pressure gradient term MS'})
        ds['WD'+dir_] = ds['WD'+dir_].assign_attrs({'long_name':Dir + ' Wind term MS'})
        
        ds['sigma'+dir_] = ds['sigma'+dir_].assign_attrs({'long_name':Dir+r' $\Sigma$'})
        ds['S'+dir_] = ds['S'+dir_].assign_attrs({'long_name':Dir+r' Residual'})
        
        ds['B'+dir_+'_acc'] = ds['B'+dir_+'_acc'].assign_attrs({'long_name':Dir+r' Lagrangian acceleration balanced signal contribution'})
        ds['B'+dir_+'_cor'] = ds['B'+dir_+'_cor'].assign_attrs({'long_name':Dir+r' Coriolis acceleration balanced signal contribution'})
        ds['B'+dir_+'_ggd'] = ds['B'+dir_+'_ggd'].assign_attrs({'long_name':Dir+r' Pressure gradient term balanced signal contribution'})
        ds['B'+dir_+'_wd'] = ds['B'+dir_+'_wd'].assign_attrs({'long_name':Dir+r' Wind term balanced signal contribution'})
        
        ds['E'+dir_+'_acc'] = ds['E'+dir_+'_acc'].assign_attrs({'long_name':Dir+r' Lagrangian acceleration residual contribution'})
        ds['E'+dir_+'_cor'] = ds['E'+dir_+'_cor'].assign_attrs({'long_name':Dir+r' Coriolis acceleration residual contribution'})
        ds['E'+dir_+'_ggd'] = ds['E'+dir_+'_ggd'].assign_attrs({'long_name':Dir+r' Pressure gradient term residual contribution'})
        ds['E'+dir_+'_wd'] = ds['E'+dir_+'_wd'].assign_attrs({'long_name':Dir+r' Wind term residual contribution'})
        
        
        ds['X'+dir_+'_acc_cor'] = ds['X'+dir_+'_acc_cor'].assign_attrs({'long_name':Dir+r' Inertial balance contribution'})
        ds['X'+dir_+'_acc_ggd'] = ds['X'+dir_+'_acc_ggd'].assign_attrs({'long_name':Dir+r' Cyclostrophique contribution'})
        ds['X'+dir_+'_acc_wd'] = ds['X'+dir_+'_acc_wd'].assign_attrs({'long_name':Dir+r' Lagrangian - wind contribution'})
        ds['X'+dir_+'_cor_ggd'] = ds['X'+dir_+'_cor_ggd'].assign_attrs({'long_name':Dir+r' Geostrophic contribution'})
        ds['X'+dir_+'_cor_wd'] = ds['X'+dir_+'_cor_wd'].assign_attrs({'long_name':Dir+r' Coriolis - wind contribution'})
        ds['X'+dir_+'_ggd_wd'] = ds['X'+dir_+'_ggd_wd'].assign_attrs({'long_name':Dir+r' Pressure gradient - wind contribution'})
        
        ds['D'+dir_+'_cyclo'] = ds['D'+dir_+'_cyclo'].assign_attrs({'long_name':Dir+r' cyclonic contribution'})
        ds['D'+dir_+'_anticyclo'] = ds['D'+dir_+'_anticyclo'].assign_attrs({'long_name':Dir+r' anticyclonic contribution'})
    return ds

def compute_mean_square_groupby(df, groupby = 'time_to_swot_1h', dirname = ('e', 'n'), compute_error = 'no', vars_errors=None):
    """ Compute closure stats on bins
    ds : dataset containing terms values for all row_numbers
    groupby : str, name of the binning variable
    dirname : directions of the reconstruction
    bootstrap :  bool, rather to compute bootstrap errors or not (longer with)
    """
    var = ['acc', 'cor', 'ggd', 'wd','sum']
    VAR = ['ACC', 'COR', 'GGD', 'WD', 'S']
    if 'meancor'+dirname[0] in df.columns :
        print('ok')
        var += ['meancor', 'meanggd']
        VAR += ['meancor'.upper(), 'meanggd'.upper()]
        closure_vars_ = closure_vars +['MEANCOR*', 'MEANGGD*']
        print('geostrophic mean corrections')

    dff = (df[[v+dirname[1] for v in var] + [v+dirname[0] for v in var]]**2).rename(columns = {var[i]+dirname[1] : VAR[i]+dirname[1] for i in range(len(var))}).rename(columns = {var[i]+dirname[0] : VAR[i]+dirname[0] for i in range(len(var))})/U2
    dff = pd.concat([dff, df[groupby]], axis=1)
    
    nb_coloc = df.set_index(groupby).groupby(groupby, observed=False)['acc'+dirname[0]].count()

    #Balanced and error
    for direction in [dirname[0], dirname[1]]:
        dff['sigma'+direction] = dff['ACC'+direction] + dff['COR'+direction] + dff['GGD'+direction] + dff['WD'+direction]
        for v in ['acc', 'cor', 'ggd', 'wd'] : 
                dff['B'+direction+'_'+v] = -((df[v+direction]*df['exc'+direction+'_'+v]))/U2
                dff['E'+direction+'_'+v] = ((df[v+direction]*df['sum'+direction]))/U2
    #pairs contributions
    for v in [v for v in df if 'prod' in v]:
        dff[v.replace('prod', 'X')]=-2*df[v]/U2

    # Cyclo/anticyclo contribution
    for direction in [dirname[0], dirname[1]]:
        dff['D'+direction+'_cyclo'] = (-2 *(df['acc'+direction]+df['cor'+direction])*df['ggd'+direction])/U2
        dff['D'+direction+'_anticyclo'] = (-2 *(df['acc'+direction]+df['ggd'+direction])*df['cor'+direction])/U2

    #Sum of both direction
    for v in closure_vars :
        dff[v.replace('*', '')] = dff[v.replace('*', dirname[0])] + dff[v.replace('*', dirname[1])]

    closure_vars_2D = [v.replace('*', dirname[0]) for v in closure_vars] + [v.replace('*', dirname[1]) for v in closure_vars] + [v.replace('*', '')for v in closure_vars]
    if 'meancor'+dirname[0] in df.columns :
        closure_vars_2D += ['MEANCORe', 'MEANGGDe', 'MEANCORn', 'MEANGGDn']
    
    if isinstance(groupby, str) : grp = [groupby]
    else : grp = groupby
    
    # centrallimit errors
    if compute_error == 'centrallimit' : 
        # Add errors central limit
        centrallimit = 2*dff.set_index(groupby).groupby(groupby, observed=False)[closure_vars_2D].std().div(np.sqrt(nb_coloc), axis=0) # 95%
        centrallimit = centrallimit.rename(columns = {v:'ser__'+v for v in closure_vars_2D})

    
    # bootstrap errors
    if compute_error == 'bootstrap' : 
        def mean_df(df):
            return df.mean()
        from scipy.stats import bootstrap
        def compute_bootstrap_error(dff):
            # print(len(dff))
            if len(dff) < 3:
                return np.nan
            else:
                data = (dff,)  # samples must be in a sequence
                return bootstrap(data, statistic=mean_df).standard_error
        #print(vars_errors)
        if vars_errors is None : vars_errors = closure_vars_2D
        #import dask.dataframe as dd
        #dfd = dd.from_pandas(dff, chunksize=10)
        DF = []
        #print(vars_errors)
        for v in vars_errors:
            DF.append(
                dff.reset_index()[vars_errors + grp]#dfd
                .groupby(groupby, observed=False)[v]
                .apply(compute_bootstrap_error)
                #.compute()
            )
            print(v)
        booterrors = pd.concat(DF, axis=1)
        booterrors = booterrors.rename(columns={v: "ber__" + v for v in booterrors.columns})
        # sum of both dim
        for v in vars_errors : 
            booterrors['ber__'+v.replace('*', '')] = booterrors['ber__'+v.replace('*', dirname[0])] + booterrors['ber__'+v.replace('*', dirname[1])]
    
    #Final steps
    dff = dff.set_index(groupby)[closure_vars_2D].groupby(groupby, observed=False).mean()
    dff['nb_coloc']= nb_coloc

    for v in closure_vars : 
        dff[v.replace('*', '')] = dff[v.replace('*', dirname[0])] + dff[v.replace('*', dirname[1])]
    
    if compute_error == 'centrallimit' : 
        dff = pd.concat([dff, centrallimit], axis=1)
    if compute_error == 'bootstrap' :
        dff = pd.concat([dff, booterrors], axis=1)
        
        
    dss = dff.to_xarray()

    #Mean geostrophics corrections :
    if 'meancor'+dirname[0] in df.columns :
        for dir_ in dirname:
            for v in ['cor', 'ggd']:
                dss[v.upper() + dir_] = dss[v.upper() + dir_] + dss['MEAN'+ v.upper() + dir_]
                dss['B'+ dir_+'_'+v] = dss['B'+ dir_+'_'+v] + dss['MEAN'+ v.upper() + dir_]
        dss['X'+ dir_+'_cor_ggd'] = dss['X'+ dir_+'_cor_ggd'] + 2*dss['MEAN'+ v.upper() + dir_]
        for v in ['cor', 'ggd']:
            dss[v.upper()] =  dss[v.upper() + dirname[0]]+ dss[v.upper() + dirname[1]]
            dss['B_'+v] = dss['B'+dirname[0]+'_'+v] + dss['B'+dirname[1]+'_'+v]
            dss['X_cor_ggd'] = dss['X'+dirname[0]+'_cor_ggd'] + dss['X'+dirname[1]+'_cor_ggd']

    dss = assign_attrs(dss, dirname)
    #dss.to_netcdf(os.path.join(zarr_dir, 'binned_diag', '_'.join(grp) + '.png'))

    return dss


def compute_variance_groupby(df, groupby = 'time_to_swot_1h', dirname = ('e', 'n'), compute_error = 'no', vars_errors=None):
    """ Compute closure stats on bins
    ds : dataset containing terms values for all row_numbers
    groupby : str, name of the binning variable
    dirname : directions of the reconstruction
    bootstrap :  bool, rather to compute bootstrap errors or not (longer with)
    """
    var = ['acc', 'cor', 'ggd', 'wd','sum']
    VAR = ['ACC', 'COR', 'GGD', 'WD', 'S']

    nb_coloc = df.set_index(groupby).groupby(groupby, observed=False)['acc'+dirname[0]].count()

    #remove Mean
    dfm = (df.set_index(groupby)[[v+dirname[1] for v in var] + [v+dirname[0] for v in var]] - df.set_index(groupby)[[v+dirname[1] for v in var] + [v+dirname[0] for v in var]].groupby(groupby).mean()).reset_index()
    for dir_ in dirname :
        dfm['sum'+dir_] = sum([dfm[v+dir_] for v in var])

        # simple var + sum- one term
        for v in var :
            dfm['exc'+dir_+'_'+v] = dfm['sum'+dir_]-dfm[v+dir_]
                
        #2 by 2 product    
        import itertools  
        l = {'acc':'acc'+dir_, 'cor':'cor'+dir_, 'ggd':'ggd'+dir_, 'wd':'wd'+dir_}
        couple = list(itertools.combinations(list(l.keys()),2))
        for c in couple : 
            dfm['prod'+dir_+'_'+'_'.join(c)] = dfm[c[0]+dir_] * dfm[c[1]+dir_]

    # MS = var as mean are null
    dss = compute_mean_square_groupby(dfm, groupby, dirname, compute_error, vars_errors)
    
    return dss


def remove_dt_traj_limit_coloc(dfr, dt):
    """ 
    Remove colocations that are at the dt-time limit of a drifter trajectories, for all cycle
    Example dfr  :
    dfr = dd.read_csv(DRIFTER['nofilter'], parse_dates=['datetime', 'cycle_date'], dtype=dtypes).set_index('row_number')[['cycle_date','datetime', 'drifter_id']]
    """
    # Build test table 
    cycle_dates = dfr.cycle_date.unique()
    drifter_id = dfr.drifter_id.unique()

    test_tmin = xr.DataArray(np.full((len(drifter_id),len(cycle_dates)), True), coords={'drifter_id':drifter_id, 'cycle_date':cycle_dates}).rename('drifter_tmin')
    test_tmax = xr.DataArray(np.full((len(drifter_id),len(cycle_dates)), True), coords={'drifter_id':drifter_id, 'cycle_date':cycle_dates}).rename('drifter_tmax')

    for d in drifter_id :
        for c in cycle_dates:
            test_tmin.loc[d, c] = (c-pd.Timedelta('10d')> tmin.loc[d])
            test_tmax.loc[d, c] = (c+pd.Timedelta('10d')< tmax.loc[d])
    dft = pd.concat([test_tmin.to_dataframe(), test_tmax.to_dataframe()], axis=1)
    dft['test'] = dft.drifter_tmin & dft.drifter_tmax

    return dfr.reset_index().set_index(['drifter_id', 'cycle_date']).where(dft.test).dropna().reset_index().set_index('row_number')

    
""" 
_________________________________________
---- PLOTS ----
_________________________________________
"""

def synthetic_figure(df, ax, xlim=[1], aviso=False, dir = 'e'):
    from cstes import U2, c0

    plt.rcParams["axes.edgecolor"] = "w"
    a = 1.5
    bbox = dict(facecolor="w", alpha=0.8, edgecolor="w")

    ts = df["sigma"+dir]
    print(ts)
    # gap between bars for readability
    if len(xlim)!=2:
        b = ts / 400
    else:
        b = xlim[0] / 400

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
    if len(xlim)!=2:
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

def plot_error(df, x, v, ax, suf = 'ber__'):
    ax.fill_between(
        df[x], df[v] - df[suf + v], df[v] + df[suf + v], color="silver"
    )

def plot_join_pdfs(ds, x, y, binx=100, biny=100):
    nsamples, xx, yy = np.histogram2d(ds[x], ds[y], bins=(binx, biny))
    da = xr.DataArray(data=nsamples, dims=[x, y],coords={x:([x], (xx[:-1] + xx[1:])/2), y:([y], (yy[:-1] + yy[1:])/2)})
    
    fig = plt.figure(figsize=(9, 4))
    
    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )
    
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    

    
    da.plot(ax=ax, add_colorbar=False)
    ax.plot(xx, -xx, color='r')
    ds[x].plot.hist(bins=binx, density=True, ax=ax_histx, zorder=1)
    ds[y].plot.hist(
        bins=biny,
        density=True,
        ax=ax_histy,
        orientation="horizontal",
        zorder=1,
    )


    # no labels
    ax.grid(zorder=0)
    ax_histx.grid()
    ax_histy.grid()
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_title('')
    ax_histx.set_title('')
    fig.tight_layout(rect=[0, 0, 1, 1])  # left, bottom, right, top (default is 0,0,1,1)
    

def select_row(df, pass_number, cycle_number, drifter_id):
    return df.where((df.pass_number==pass_number)&(df.cycle_number==cycle_number)&(df.drifter_id==drifter_id)).dropna()

""" 
_________________________________________
---- ONE COLOC PLOTS ----
_________________________________________
"""

def plot_exemple(row_number, df, directory, swot_product = '250m'):
    """
    Parameters :
    ------------
        row_number : index of the coloc to find in df
        df : dataframe, should contains : pass_number, cycle_number, longitude, latitude, drifter_id, drifter_type, acce, accn, core, corn, ggde, ggdn, wde, wdn,sume, sumn
    
    """
    # Collect info from row_number
    cycle_number = df.loc[row_number].cycle_number
    pass_number = df.loc[row_number].pass_number
    drifter_id = df.loc[row_number].drifter_id

    # select good rows in df
    df_ = df.where((df.pass_number==pass_number)&(df.cycle_number==cycle_number)&(df.drifter_id==drifter_id)).dropna()#all
    dfc_ = select_nearest_swot_coloc(df_)#nearest

    #
    dl = 0.2
    bbox = [dfc_.longitude.values[0]-dl, dfc_.longitude.values[0]+dl, dfc_.latitude.values[0]-dl, dfc_.latitude.values[0]+dl]
    #print(bbox)

    #SWOT data
    if swot_product=='250m': dfs = browse_swot_250().reset_index()
    if swot_product=='2km': dfs = browse_swot_2km().reset_index()  
    dss = xr.open_dataset(dfs.where((dfs.pass_number==pass_number)&(dfs.cycle_number==cycle_number)).dropna().file.values[0])
    dss = dss.where((dss.latitude>bbox[2]) & (dss.latitude<bbox[3])&(dss.longitude>bbox[0]) & (dss.longitude<bbox[1]))
    
    #ERA data
    era = xr.open_dataset('/Users/mdemol/DATA_WIND/era5/adaptor.mars.internal-1726002205.1540956-13738-7-c8d88fd1-3ec7-4113-8790-c92c59938aa6.nc')
    
    #PLOT
    fig = plt.figure( frameon=False, figsize=(20,12))

    ax = fig.add_subplot(241)
    df_['sume'] = df_.acce + df_.core + df_.ggde + df_.wde
    df_ = df_.set_index('datetime')
    df_.acce.plot(ax=ax, label = 'acc', c=c0['acc'], ls='', marker='.')
    df_.core.plot(ax=ax, c=c0['cor'], label = 'cor', ls='', marker='.')
    df_.ggde.plot(ax=ax,c=c0['ggd'],  label = 'ggd', ls='', marker='.')
    df_.wde.plot(ax=ax,c=c0['wd'],  label = 'wd', ls='', marker='.')
    (-(df_.acce + df_.core + df_.wde)).plot(ax=ax,c=c0['ggd'],  label = 'gge for balance', ls='--')
    df_.sume.plot(ax=ax, c='k', label = 's', ls='--')
    ax.axvline(dfc_.datetime.values[0], color = 'r',ls=':')
    ax.axvline(pd.to_datetime(dss.time.mean().values), color = 'b',ls=':')
    ax.legend()
    ax.grid()
    ax.set_title('East-West terms')
    #ax.set_ylim(-7e-5, 7e-5)


    ax = fig.add_subplot(245)
    df_['sumn'] = df_.accn + df_.corn + df_.ggdn + df_.wdn
    df_.accn.plot(ax=ax, label = 'acc', c=c0['acc'], ls='', marker='.')
    df_.corn.plot(ax=ax, c=c0['cor'], label = 'cor', ls='', marker='.')
    df_.ggdn.plot(ax=ax,c=c0['ggd'],  label = 'ggd', ls='', marker='.')
    df_.wdn.plot(ax=ax,c=c0['wd'],  label = 'wd', ls='', marker='.')
    (-(df_.accn + df_.corn + df_.wdn)).plot(ax=ax,c=c0['ggd'],  label = 'ggn for balance', ls='--')
    df_.sumn.plot(ax=ax, c='k', label = 's', ls='--')
    ax.axvline(dfc_.datetime.values[0], color = 'r',ls=':')
    ax.axvline(pd.to_datetime(dss.time.mean().values), color = 'b',ls=':')
    ax.legend()
    ax.grid()
    ax.set_title('North-south terms')
    ax.set_title('Global view')
    #ax.set_ylim(-7e-5, 7e-5)

    # SWOT ETA global view
    ax = fig.add_subplot(242, projection=ccrs.Orthographic(df.longitude.mean(), df.latitude.mean()))
    ax.add_feature(cfeature.LAND,)
    gl = ax.gridlines(draw_labels=True,)
    bbox_all = [1, 6, 37, 43.5]
    ax.set_extent(bbox_all)
    dss['eta'] = dss.cvl_mean_dynamic_topography_cnes_cls_22 + dss.cvl_ocean_tide_fes_2022 + dss.duacs_ssha_karin_2_filtered
    dss.where(dss.duacs_editing_flag==0).eta.plot(x='longitude', y='latitude', ax=ax, transform=crs,cmap='viridis')
    df_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, s=2 )
    dfc_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, marker='*', color='r', s=10 )
    
    # SWOT ETA + ERA wind
    ax = fig.add_subplot(244, projection=ccrs.Orthographic(df.longitude.mean(), df.latitude.mean()))
    ax.add_feature(cfeature.LAND,)
    gl = ax.gridlines(draw_labels=True,)
    ax.set_extent(bbox)
    dss['eta'] = dss.cvl_mean_dynamic_topography_cnes_cls_22 + dss.cvl_ocean_tide_fes_2022 + dss.duacs_ssha_karin_2_filtered
    #dss.where(dss.duacs_editing_flag==0).eta.plot(x='longitude', y='latitude', ax=ax, transform=crs,cmap='viridis')
    dss.eta.plot(x='longitude', y='latitude', ax=ax, transform=crs,cmap='viridis')
    df_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, s=2 )
    dfc_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, marker='*', color='r', s=10 )
    ax.set_title('SWOT MDT + SLA + tides correction')

    t = dss.time.mean()
    era_ = era.sel(time=dss.time.mean(), method='nearest').sortby('latitude')
    era_ = era_.sel(longitude=slice(bbox[0], bbox[1]), latitude=slice(bbox[2], bbox[3]))
    Q = era_.plot.quiver('longitude', 'latitude', 'u10', 'v10', ax=ax, transform =crs, scale=250, color='magenta', clip_on = False, add_guide = False)


    # SWOT MDT
    ax = fig.add_subplot(243, projection=ccrs.Orthographic(df.longitude.mean(), df.latitude.mean()))
    ax.add_feature(cfeature.LAND,)
    gl = ax.gridlines(draw_labels=True,)
    ax.set_extent(bbox)
    dss.cvl_mean_dynamic_topography_cnes_cls_22.plot(x='longitude', y='latitude', ax=ax, transform=crs,cmap='viridis')
    df_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, s=2 )
    dfc_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, marker='*', color='r', s=10 )
    ax.set_title('SWOT MDT')

    # SWOT sigma0
    ax = fig.add_subplot(246, projection=ccrs.Orthographic(df.longitude.mean(), df.latitude.mean()))
    ax.add_feature(cfeature.LAND,)
    gl = ax.gridlines(draw_labels=True,)
    ax.set_extent(bbox)
    dss.sig0_karin_2.plot(x='longitude', y='latitude', ax=ax, transform=crs,cmap='viridis')
    df_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, s=2 )
    dfc_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, marker='*', color='r', s=10 )
    ax.set_title('SWOT sigma0')

    # SWOT velocities
    ax = fig.add_subplot(247, projection=ccrs.Orthographic(df.longitude.mean(), df.latitude.mean()))
    ax.add_feature(cfeature.LAND,)
    gl = ax.gridlines(draw_labels=True,)
    ax.set_extent(bbox)
    dss['U'] = np.sqrt(dss.duacs_speed_zonal**2 + dss.duacs_speed_meridional**2)
    dss.U.plot(x='longitude', y='latitude', ax=ax, transform=crs,cmap='viridis', vmax=5, vmin=0)
    df_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, s=2 )
    dfc_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, marker='*', color='r', s=10 )
    ax.set_title('SWOT geostrophic velocity')

    # SWOT duacs editing flags
    ax = fig.add_subplot(248, projection=ccrs.Orthographic(df.longitude.mean(), df.latitude.mean()))
    ax.add_feature(cfeature.LAND,)
    gl = ax.gridlines(draw_labels=True,)
    def get_indice(flag): 
        if np.isnan(flag):
            return np.nan
        else :
            return flag_indices[flag]
    editing_flags = xr.apply_ufunc(get_indice, dss.duacs_editing_flag, vectorize=True)
    cm, fmt, tickz, norm = create_flag_cmap()
    im = dss.duacs_editing_flag.plot(x='longitude', y='latitude', transform =crs, ax=ax, cmap=cm, norm=norm,)
    plot_flag_colorbar(fig, im, fmt, tickz)
    df_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, s=2 )
    dfc_.plot.scatter('longitude', 'latitude', transform =crs, ax=ax, marker='*', color='r', s=10 )
    ax.set_extent(bbox)
    ax.set_title('SWOT editing flags')

    #ax.set_ylim(-7e-5, 7e-5)

    fig.suptitle(f'row number = {row_number}, pass_number = {pass_number}, cycle = {cycle_number}, drifter = {dfc_.drifter_type.values}, {drifter_id} \n')#+ f'wd ={np.sqrt(dfwc_.u10**2 + dfwc_.v10**2)}')
    fig.tight_layout()
    fig.savefig(os.path.join(images_dir, directory, f'{row_number}.png'), dpi=200, bbox_inches='tight')


"""
EDITING_FLAGS PLOT
________________
"""

flag_values = [0, 5, 10, 20, 30, 50, 70, 100, 101, 102, 200]
flag_legend = ['good','local_outliers', 'bad_quality_coast','ice','soft_outliers',  'extremes', 'mission_events', 'bad_swath_extremities', 'not_on_sea', 'no_data', 'gradient_nan', ]
flag_color = ['pink', 'coral', 'orange', 'lightblue', 'magenta', 'red', 'blue','green', 'yellow', 'grey', 'darkgrey']
len_lab = len(flag_values)
flag_indices = {flag_values[i] : i for i in range(len(flag_values))}

def create_flag_cmap():
    from matplotlib.colors import ListedColormap
    import matplotlib
    cm = ListedColormap(flag_color)
    norm_bins = np.arange(len_lab)+1
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: flag_legend[norm(x)])
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    return cm, fmt, tickz, norm
    
def plot_flag_colorbar(fig, im, fmt, tickz):
    cb = im.colorbar   
    cb.remove()
    cb = fig.colorbar(im, format=fmt, ticks=tickz)


"""
COM
________________
"""
def MSnoiseggd_from_sshnoisestd(sigmaN, d=2e3) : 
    """ 
    sigmaN : white noise std in m
    d : grid spacing in m
    """
    print('Caution : std must be given in m, E_ggd given in gamma^2')
    from cstes import U2
    g=9.81
    K = -sum(np.arange(0, 9)*[1/280, -4/105, 1/5, 4/5, 0, -4/5, -1/5, 4/105, -1/280])
    b=(1/280)**2 + (4/105)**2+ (1/5)**2+(4/5)**2
    return 4*g**2/(K**2)/(d**2)*b*(sigmaN**2)/U2

def sshnoisestd_from_MSnoiseggd(E_ggd, d=2e3) : 
    """ 
    E_ggd : pressure gradient MS in gamma^2
    d : grid spacing in m
    """
    print('Caution : E_ggd must be given in gamma^2, std must return in m')
    from cstes import U2
    g=9.81
    K = -sum(np.arange(0, 9)*[1/280, -4/105, 1/5, 4/5, 0, -4/5, -1/5, 4/105, -1/280])
    b=(1/280)**2 + (4/105)**2+ (1/5)**2+(4/5)**2
    
    return np.sqrt(E_ggd*U2*(K*d)**2/(4*g**2*b))



