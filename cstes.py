""" MAIN CSTES/PATHS ETC
---------------------------
"""



"""
PATHS
---------------------------------------------------------------------------------------------------------
"""
#DATARMOR
#swot_dir = '/home/datawork-lops-osi/aponte/swot/cswot/swot/l3_v1.0.1'
#drifters_dir = '/home/datawork-lops-osi/aponte/swot/cswot/drifters_harmonized'
#zarr_dir = "/home/datawork-lops-osi/aponte/margot/med_coloc"

#LOCAL MARGOT
swot_dir = '/Users/mdemol/DATA_KARIN/L3_250'
drifters_dir = '/Users/mdemol/DATA_DRIFTERS/drifters_harmonized'
zarr_dir = "/Users/mdemol/DATA_MED_COLOC"
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

