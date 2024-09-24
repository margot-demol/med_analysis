import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import os
from glob import glob


def find_combination(df):
    id_comb = []
    # Alti
    ggrade, ggradn = [v for v in df if 'ggrade' in v], [v for v in df if 'ggradn' in v]
    alti_var = [v.replace('ggrade_','') for v in ggrade]
    
    # Wind
    winde, windn = [v for v in df if 'winde' in v], [v for v in df if 'windn' in v]
    wind_var = [v.replace('winde','').replace('_', '') for v in winde]

    import itertools  

    couple = list(itertools.product(alti_var, wind_var))
    E, N = list(itertools.product(ggrade, winde)), list(itertools.product(ggradn, windn))
    
    id_comb = dict()
    for i in range(len(E)):
        le = {"acc" : "acce", "coriolis" : "coriolise", "ggrad":E[i][0], "wind":E[i][1]}
        ln = {"acc" : "accn", "coriolis" : "coriolisn", "ggrad":N[i][0], "wind":N[i][1]}
        id_comb['__'.join(couple[i])] = le, ln
    return id_comb


def create_DS(df, id_comb):
    D = []
    for comb in id_comb :
        coords_list = ['datetime', 'longitude', 'latitude', 'pass_number','time_to_swot','cycle_number','drifter_id', 'drifter_type']
        ds = df[coords_list].to_xarray()
        import itertools
        ds['id_comb'] = comb
        
        for direction in ['e', 'n'] : 
            if direction == 'e' : l = id_comb[comb][0]
            if direction == 'n' : l = id_comb[comb][1]
                
            # sum 
            ds['sum'+direction] = sum([df[v] for v in l.values()])

            # simple var + sum- one term
            for v in list(l.keys()) : 
                ds[v+direction] = df[l[v]]
                ds['exc'+direction+'_'+v] = ds['sum'+direction]-ds[v+direction]
                
            #2 by 2 product    
            couple = list(itertools.combinations(list(l.keys()),2))
            for c in couple : 
                ds['prod'+direction+'_'+'_'.join(c)] = ds[c[0]+direction] * ds[c[1]+direction]
                
        D.append(ds)
    DS = xr.concat(D, dim='id_comb').set_coords(coords_list)
    return DS
