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
space = 'local'
if space == 'datarmor' :
    #DATARMOR
    swot_dir = '/Users/mdemol/DATA_KARIN/L3_unsmoothed_calval_1.0.2'
    swot_dir_2km = '/Users/mdemol/DATA_KARIN/L3_calval_1.0.2'
    swot_dir = '/home/datawork-lops-oc/aponte/swot/cswot/swot/L3_unsmoothed_calval_1.0.2'
    drifters_dir = '/home/datawork-lops-oc/aponte/swot/cswot/drifters_harmonized'
    zarr_dir = "/home/datawork-lops-oc/aponte/margot/med_coloc"

if space == 'local':
    #LOCAL MARGOT
    swot_dir = '/Users/mdemol/DATA_KARIN/L3_unsmoothed_calval_1.0.2'
    swot_dir_2km = '/Users/mdemol/DATA_KARIN/L3_calval_1.0.2'
    drifters_dir = '/Users/mdemol/DATA_DRIFTERS/drifters_harmonized'
    zarr_dir = "/Users/mdemol/DATA_MED_COLOC"
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
err_acc = err_acc = [175606, 175607, 175608, 175609, 175610, 175611, 175612, 175613, 175614,
       175615, 175616, 175617, 175618, 175619, 175620, 175621, 175622, 175623,
       175624, 175625, 175626, 175627, 175628, 175629, 175630, 175631, 175632,
       175633, 175634, 175635, 175636, 175637, 175638, 175639, 175640, 175641,
       175642, 175643, 175644, 175645, 175646, 175647, 175648, 175649, 175650,
       175651, 175652, 175653, 175654, 175655, 175656, 175657, 175658, 175659,
       175660, 175661, 175662, 175663, 175664, 175665] + [14980, 14981, 14982, 14983, 14984, 14985, 14986, 14987, 14988, 14989,
       14990, 14991, 14992, 14993, 14994, 14995, 14996, 14997, 14998, 14999,
       15000, 15001, 15002, 15003, 15004] + [295158, 295159, 295160, 295161, 295162, 295163, 295164, 295165, 295166,
       295167, 295168, 295169, 295170, 295171, 295172, 295173, 295174, 295175,
       295176, 295177, 295178, 295179, 295180, 295181, 295182]
# Outliers in swot 250 (south of Minorca)
idpbswot = ['300534062472380', '300534062474750', '4388589']
tminpbswot = [pd.to_datetime('2023-04-27 23:40:00'), pd.to_datetime('2023-05-06 06:20:00'),pd.to_datetime('2023-05-24 13:30:00')]
tmaxpbswot = [pd.to_datetime('2023-04-29 02:40:00'),pd.to_datetime('2023-05-25 03:10:00'),pd.to_datetime('2023-05-27 14:40:00')]
def remove_pb_swot(df):
    for i in range(len(idpbswot)):
        pb = (df.drifter_id ==idpbswot[i]) & (df.datetime>=tminpbswot[i]) & (df.datetime<=tmaxpbswot[i])
        df = df[~pb]
        return df
