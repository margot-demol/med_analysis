""" MAIN CSTES/PATHS ETC
---------------------------
"""
U = 9.81*1e-6
U2 = U**2

c0 ={'acc':'#941717', 'cor':'#388E3C','ggd':'#42A5F5', 'wd':'#FFA000'}

"""
PATHS
---------------------------------------------------------------------------------------------------------
"""
space = 'datarmor'
version_swot = '2.0.1'
version_drifter = 'v1'


if space == 'datarmor' :
    #DATARMOR
    swot_dir_2km = f'/home/datawork-lops-oc/aponte/swot/cswot/swot/L3_calval_{version_swot}'
    swot_dir = f'/home/datawork-lops-oc/aponte/swot/cswot/swot/L3_unsmoothed_calval_{version_swot}'
    drifters_dir = '/home/datawork-lops-oc/aponte/swot/cswot/drifters_harmonized'
    zarr_dir = f"/home/datawork-lops-oc/aponte/margot/DATA_MED_COLOC_v{version_swot}"
    images_dir = '/Users/mdemol/ownCloud/PhD/images/med_images'

if space == 'local':
    #LOCAL MARGOT
    swot_dir = '/Users/mdemol/DATA_KARIN/L3_unsmoothed_calval_1.0.2'
    swot_dir_2km = '/Users/mdemol/DATA_KARIN/L3_calval_1.0.2'
    drifters_dir = '/Users/mdemol/DATA_DRIFTERS/drifters_harmonized'
    zarr_dir = "/Users/mdemol/DATA_MED_COLOC_v{version_swot}"
    images_dir = '/Users/mdemol/ownCloud/PhD/images/med_images'
    #path_wind = 

color = {
    'BIOSWOT: CARTHE': "firebrick",
    'FAST-SWOT: CARTHE':'darkred',
    'C-SWOT: CARTHE': "orange",
    'BIOSWOT: CODE': "green",
    'BIOSWOT: SVPOGS': "darkblue",
    'BIOSWOT: SVPSIO': "teal",
    'C-SWOT: SVP': "lightblue",
    'C-SWOT: SPOTTER': "yellow",
    'C-SWOT: MELODI': "hotpink",
    'FAST-SWOT: HEREON':'peru',
    'FAST-SWOT: OSMC':'violet',
}

surface_drifters = ['CARTHE', 'CODE', 'HEREON', 'MELODI', 'SPOTTER']
depth_drifters = ['OSMC','SVP','SVPSIO', 'SVP-B','SVPOGS', 'SVPSIO', 'SVPBGC']

depth_50 = ['300534062472380','300534060112360','300534060113380']
depth_100 = ['300534060116350','300534060017750']

drifters_sources = 'all_med_variational_10min_v1.nc'

drifters_structure = {
    'Dipole D1': ['300534060113380','300534062474750', '300534060112360', '0-4388605'],
    'Cyclone C1' : ['300534060112360'], 
    'Cyclone C2' : ['300534060112360'],#, '4389458', '4388938', '300534060015760'], 
    'Anticyclone A3':[ '300534060116350', '300534060315840', '300534062470440'], 
    'Cyclone C3' : ['300534060017440','300534061170360', '300534060015760','300534062479690', '4388589', '0-4388635', '13'],
    'BioSWOT-Med' : [ '300534060216070',
 '300534062472690', '300534060218400', '0-4388581',
 '4388589', '4388593', '8', '300534062471390', '13', '12',
 '4388625', '7', '300534064107880', '4388553', '14', '300534064107900',
 '4388634', '4388587', '300534064109870', '4388632', '6', '4388559', '4388640',
 '300534064109890', '4388636', '300534062479690', '9', '300534064108950', '10',
 '4389458', '300534064300460', '4389089', '300534061395970', '4388942',
 '300534064300510', '4388943', '147', '4389467', '4388938', '300534061395980',
 '300534061398910', '4388923', '300534061395960', '300534061398320', '4389009',
 '4388926', '300534061398370' ,'300534061492130', '4450524', '4450435',
 '300534064103890', '4450510', '4450425', '4450433', '4450430', '4450431',
 '300534064104890', '300534064104920', '4450432', '4450426' ,'300534064106800',
 '300534064103920', '17', '4450525', '4389439', '0-4367706', '300534061170360',
 '300534060315840', '300534060211410'], 
    'BioSWOT-Med anticyclone A1' : ['300534060216070',
                       '300534062472690', 
                       '300534060218400',
                       '4388589', 
                       '300534062471390', 
                       '4388625',
                       '4388587',
                       '300534064109870',
                       '6',
                       '4388559',
                       '300534064109890', 
                       '4388636',
                       '9',
                       '10',
                       '4389458',
                       '300534064300460',
                       '4389089', 
                       '300534061395970',
                       '4388942',
                       '300534064300510',
                       '4388943',
                       '147', 
                       '4389467', 
                       '4388938',
                       '300534061395980',
                       '300534061398910',
                       '4388923',
                       '300534061395960',
                       '300534061398320',
                       '4389009',
                       '4388926', 
                       '300534061398370' ,
                       '300534061492130', 
                    ],
    'BioSWOT-Med front F1' : ['0-4388581','4388593','8','12','13','7','300534064107880','4388553','300534064107900','4388634','4388632', '4388640','300534062479690','300534064108950', '4450524','4450435','300534064103890', '4450510','4450425','4450433','4450430','4450431',
 '300534064104890', '300534064104920', '4450432', '4450426' ,'300534064106800',
 '300534064103920', '17', '4450525', '4389439', '0-4367706', '300534061170360',
 '300534060315840', '300534060211410'], 
    'FaSt-SWOT anticyclone A2' : ['4694122', '4694103', '6204605' ,'4694040', '279', '6204607', '280' ,'278',
 '6204608', '6204606', '6204604', '286', '284', '4694198', '281', '287', '288',
 '283', '4694188', '285', '4694203', '4694193', '4694201', '4694199', '4694189',
 '289', '282', '4694206', '4694461', '4694223', '4694458',]

}

structure = {'Dipole D1':dict(cmin=505, lomin = 4, lamin = 39.2, cmax = 516, lomax = 5.15, lamax=40, pass_=3,),
          'Cyclone C1':dict(cmin=517, lomin = 3.9, lamin = 39.3, cmax = 520, lomax = 4.25, lamax=39.75, pass_=3,),
            'Cyclone C2':dict(cmin=529, lomin = 4.25, lamin = 38.5, cmax = 542, lomax = 4.75, lamax=39.25, pass_=3,),
          'Anticyclone A3':dict(cmin=479, lomin = 4.5, lamin = 41.9, cmax = 493, lomax = 5.1, lamax=42.3, pass_=3,),
          'Cyclone C3':dict(cmin=499, lomin = 4, lamin = 40.1, cmax = 510, lomax = 4.8, lamax=40.65, pass_=3,),
             'BioSWOT-Med' : dict(cmin=500, lomin = 4.75, lamin = 40.3, cmax = 523, lomax = 5.5, lamax=41.3, pass_=3,),
             'BioSWOT-Med front F1' : dict(cmin=500, lomin = 4.75, lamin = 40.3, cmax = 520, lomax = 5.5, lamax=41.3, pass_=3,),
             'BioSWOT-Med anticyclone A1' : dict(cmin=500, lomin = 4.75, lamin = 40.3, cmax = 523, lomax = 5.5, lamax=41.3, pass_=3,),
             'FaSt-SWOT anticyclone A2' : dict(cmin=501, lomin = 1.25, lamin = 39.5, cmax = 524, lomax = 2, lamax=40.2, pass_=16,),
         }          
for k in structure :
    structure[k].update({'drifter_id':drifters_structure[k]})



"""
PROJECTION
---------------------------------------------------------------------------------------------------------
"""
import pyproj
from rasterio.transform import Affine
import numpy as np
import pandas as pd

def get_proj(lonc, latc):  # converts from lon, lat to native map projection x,y
    """Create pyproj Proj object, project is an azimutal Eqsuidistant projection centered on the central point of the selected satellite track = matching point
    https://proj.org/operations/projections/aeqd.html

    Parameters
    ----------
    lonc,latc : float
        central longitude and latitude of the satellite track, matching point on which the box will be centered
    Return
    ------
    pyproj.Proj object
    """
    return pyproj.Proj(
        proj="aeqd", lat_0=latc, lon_0=lonc, datum="WGS84", units="m"
    )  # aeqd Azimutal EQuiDistant projection centered on lonc,latc


def lonlat2xy(lonc, latc, phi, lon, lat, lon1=None, lat1=None):
    """return coordinates with origin at (lonc, latc) and x-axis aligned
    with (lonc, latc) - (lon1, lat1) direction (lon,lat -> x-along satellite track, y-normal to satellite track)

    Parameters
    ----------
    lonc, latc, phi: float
        central position and orientation of the box
    lon, lat : np.array, np.array
        local grid of the box
    lon1,lat1 : float
        end of the satellite track

    Return
    ------
    local x, local y : np.array, np.array

    """
    proj = get_proj(lonc, latc)
    # get local coordinate
    xc, yc = proj.transform(lonc, latc)
    xl, yl = proj.transform(lon, lat)
    if phi is None:
        # compute phi from x1 and y1
        x1, y1 = proj.transform(lon1, lat1)
        # get orientation of defined by central point and point 1
        phi = np.arctan2(y1 - yc, x1 - xc) * 180 / np.pi
    # build affine operators
    a_fwrd = Affine.translation(-xc, -yc) * Affine.rotation(-phi, pivot=(xc, yc))
    # a_back = ~a_fwrd

    x, y = a_fwrd * (xl, yl)

    return x, y


"""
depth_50 = ['300534062472380','300534060112360','300534060113380']
depth_100 = ['300534060116350','300534060017750']

depth = df['pass_number'].copy()
depth.loc[df['drifter_type'].isin(surface_drifters)]=0
depth.loc[df['drifter_type'].isin(depth_drifters)]=15
depth.loc[df['drifter_id'].isin(depth_50)]=50
depth.loc[df['drifter_id'].isin(depth_100)]=100

df['depth'] = depth
"""


# TO REMOVE
# Outliers in drifters acc
err_acc = [16523,
 16524,
 16525,
 16526,
 16527,
 16528,
 16529,
 16530,
 16531,
 16532,
 16533,
 16534,
 16535,
 16536,
 16537,
 16538,
 16539,
 16540,
 16541,
 16542,
 16543,
 16544,
 16545,
 16546,
 16547,
 179757,
 179758,
 179759,
 179760,
 179761,
 179762,
 179763,
 179764,
 179765,
 179766,
 179767,
 179768,
 179769,
 179770,
 179771,
 179772,
 179773,
 179774,
 179775,
 179776,
 179777,
 179778,
 179779,
 179780,
 179781,
 179782,
 179783,
 179784,
 179785,
 179786,
 179787,
 179788,
 179789,
 179790,
 179791,
 179792,
 179793,
 179794,
 179795,
 179796,
 179797,
 179798,
 179799,
 179800,
 179801,
 179802,
 179803,
 179804,
 179805,
 179806,
 179807,
 179808,
 179809,
 179810,
 179811,
 179812,
 179813,
 179814,
 179815,
 179816,
 272459,
 272460,
 272461,
 272462,
 272463,
 272464,
 272465,
 272466,
 272467,
 272468,
 272469,
 272470,
 272471,
 272472,
 272473,
 272474,
 272475,
 272476,
 272477,
 272478,
 272479,
 272480,
 272481,
 272482,
 272483]

# Outliers in swot 250 (south of Minorca)
idpbswot = ['0-4367707', '300534062472380', '300534062474750']
tminpbswot = [pd.to_datetime('2023-04-23 18:40:00'),
              pd.to_datetime('2023-04-27 23:20:00'),
              pd.to_datetime('2023-05-05 02:40:00')]

tmaxpbswot = [pd.to_datetime('2023-04-23 18:40:00'),
              pd.to_datetime('2023-04-30 20:20:00'),
              pd.to_datetime('2023-05-25 03:10:00')]
def remove_pb_swot(df):
    for i in range(len(idpbswot)):
        pb = (df.drifter_id ==idpbswot[i]) & (df.datetime>=tminpbswot[i]) & (df.datetime<=tmaxpbswot[i])
        df = df[~pb]
    return df
