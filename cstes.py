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
depth_drifters = ['OSMC','SVP', 'SVPOGS', 'SVPSIO']

depth_50 = ['300534062472380','300534060112360','300534060113380']
depth_100 = ['300534060116350','300534060017750']


"""
PROJECTION
---------------------------------------------------------------------------------------------------------
"""
import pyproj
from rasterio.transform import Affine
import numpy as np

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
err_acc = [281205, 281206, 281207, 281208, 281209, 281210, 281211, 281212, 281213,
       281214, 281215, 281216, 281217, 281218, 281219, 281220, 281221, 281222,
       281223, 281224, 281225, 281226, 281227, 281228, 281229] + [171545, 171546, 171547, 171548, 171549, 171550, 171551, 171552, 171553,
       171554, 171555, 171556, 171557, 171558, 171559, 171560, 171561, 171562,
       171563, 171564, 171565, 171566, 171567, 171568, 171569, 171570, 171571,
       171572, 171573, 171574, 171575, 171576, 171577, 171578, 171579, 171580,
       171581, 171582, 171583, 171584, 171585, 171586, 171587, 171588, 171589,
       171590, 171591, 171592, 171593, 171594, 171595, 171596, 171597, 171598,
       171599, 171600, 171601, 171602, 171603, 171604] + [15907, 15908, 15909, 15910, 15911, 15912, 15913, 15914, 15915, 15916,
       15917, 15918, 15919, 15920, 15921, 15922, 15923, 15924, 15925, 15926,
       15927, 15928, 15929, 15930, 15931]