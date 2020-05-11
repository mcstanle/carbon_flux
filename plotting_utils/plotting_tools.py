"""
A collection of tools to accomodate the following plotting purposes -
1. locations on map
2. scale factors for a month
3. time series of fluxes

Author   : Mike Stanley
Created  : May 6, 2020
Modified : May 6, 2020

================================================================================
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_single_locs(
    lon_lat_pts, extent_lst,
    lon, lat, title=None, save_loc=None
                    ):
    """
    Plot a single points on a map

    Parameters:
        lon_lat_pts (list)   : list of (lon, lat) indices tuples
        extent_lst  (list)   : plotting region
        lon         (np arr) : 1d of array of longitudes
        lat         (np arr) : 1d of array of latitudes
        title       (str)    : title for plot if given
        save_loc    (str)    : save location of plot (default None)

    Returns:
        matplotlib plot save if location give

    Note:
    - the indices in lon_lat_pts should index the lon/lat arrays
    """
    # transform points
    lons = [lon[lon_pt] for lon_pt, lat_pt in lon_lat_pts]
    lats = [lat[lat_pt] for lon_pt, lat_pt in lon_lat_pts]

    # make the plot
    fig = plt.figure(figsize=(12.5, 8))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')

    ax.scatter(lons, lats, transform=ccrs.PlateCarree(),
               marker='s', s=50)

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.OCEAN)

    ax.set_extent(extent_lst)

    if title:
        ax.set_title(title)

    if save_loc:
        plt.savefig(save_loc)

    plt.show()
