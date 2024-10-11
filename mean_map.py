import numpy as np
import dask.dataframe as dd
import utils_validation
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

def set_cartopy(ax):
    ax.set_extent([-57, -52, -22.5, -18.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=.6)
    ax.add_feature(cfeature.STATES, linewidth=.3)
    
    # Configure gridlines and labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=.2, color='gray', alpha=0.5, linestyle='--')
    
    # Show labels only on the bottom and left
    gl.left_labels = True
    gl.bottom_labels = True
    gl.right_labels = False
    gl.top_labels = False
calculator = utils_validation.MetricsCalculator()
aplicator = utils_validation.MetricsApplicator(calculator) 


track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_mvk/*.parquet"))
concat_df_gsmap_mvk = dd.read_parquet(track_files).compute()
pps_convective_gsmap_mvk, radar_convective_gsmap_mvk, pps_nonconvective_gsmap_mvk, radar_nonconvective_gsmap_mvk, pps_all_gsmap_mvk, radar_all_gsmap_mvk= aplicator.concatenate_convec_and_nonconvec(concat_df_gsmap_mvk)
array_x_convec_gsmap_mvk, array_y_convec_gsmap_mvk, array_x_nonconvec_gsmap_mvk, array_y_nonconvec_gsmap_mvk,array_x_all_gsmap_mvk,array_y_all_gsmap_mvk = aplicator.concatenate_array_positions(concat_df_gsmap_mvk)
del(concat_df_gsmap_mvk)

print("Done")
#gsmap_nrt
track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_nrt/*.parquet"))
concat_df_gsmap_nrt = dd.read_parquet(track_files).compute()
pps_convective_gsmap_nrt, radar_convective_gsmap_nrt, pps_nonconvective_gsmap_nrt, radar_nonconvective_gsmap_nrt, pps_all_gsmap_nrt, radar_all_gsmap_nrt= aplicator.concatenate_convec_and_nonconvec(concat_df_gsmap_nrt)
array_x_convec_gsmap_nrt, array_y_convec_gsmap_nrt, array_x_nonconvec_gsmap_nrt, array_y_nonconvec_gsmap_nrt,array_y_all_gsmap_nrt,array_x_all_gsmap_nrt= aplicator.concatenate_array_positions(concat_df_gsmap_nrt)
del(concat_df_gsmap_nrt)
print("Done")

#gsmap_now
track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_now/*.parquet"))
concat_df_gsmap_now = dd.read_parquet(track_files).compute()
pps_convective_gsmap_now, radar_convective_gsmap_now, pps_nonconvective_gsmap_now, radar_nonconvective_gsmap_now, pps_all_gsmap_now, radar_all_gsmap_now= aplicator.concatenate_convec_and_nonconvec(concat_df_gsmap_now)
array_x_convec_gsmap_now, array_y_convec_gsmap_now, array_x_nonconvec_gsmap_now, array_y_nonconvec_gsmap_now,array_x_all_gsmap_now,array_y_all_gsmap_now= aplicator.concatenate_array_positions(concat_df_gsmap_now)
del(concat_df_gsmap_now)
print("Done")

#gauge mvk, gauge nrt, gauge now
track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_gauge_mvk/*.parquet"))
concat_df_gauge_mvk = dd.read_parquet(track_files).compute()
pps_convective_gauge_mvk, radar_convective_gauge_mvk, pps_nonconvective_gauge_mvk, radar_nonconvective_gauge_mvk, pps_all_gauge_mvk, radar_all_gauge_mvk= aplicator.concatenate_convec_and_nonconvec(concat_df_gauge_mvk)
array_x_convec_gauge_mvk, array_y_convec_gauge_mvk, array_x_nonconvec_gauge_mvk, array_y_nonconvec_gauge_mvk,array_y_all_gauge_mvk,array_x_all_gauge_mvk= aplicator.concatenate_array_positions(concat_df_gauge_mvk)
del(concat_df_gauge_mvk)
print("Done")

track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_gauge_nrt/*.parquet"))
concat_df_gauge_nrt = dd.read_parquet(track_files).compute()
pps_convective_gauge_nrt, radar_convective_gauge_nrt, pps_nonconvective_gauge_nrt, radar_nonconvective_gauge_nrt,pps_all_gauge_nrt, radar_all_gauge_nrt= aplicator.concatenate_convec_and_nonconvec(concat_df_gauge_nrt)
array_x_convec_gauge_nrt, array_y_convec_gauge_nrt, array_x_nonconvec_gauge_nrt, array_y_nonconvec_gauge_nrt,array_y_all_gauge_nrt,array_x_all_gauge_nrt= aplicator.concatenate_array_positions(concat_df_gauge_nrt)
del(concat_df_gauge_nrt)
print("Done")

track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_gauge_now/*.parquet"))
concat_df_gauge_now = dd.read_parquet(track_files).compute()
pps_convective_gauge_now, radar_convective_gauge_now, pps_nonconvective_gauge_now, radar_nonconvective_gauge_now, pps_all_gauge_now, radar_all_gauge_now= aplicator.concatenate_convec_and_nonconvec(concat_df_gauge_now)
array_x_convec_gauge_now, array_y_convec_gauge_now, array_x_nonconvec_gauge_now, array_y_nonconvec_gauge_now,array_x_all_gauge_now,array_y_all_gauge_now= aplicator.concatenate_array_positions(concat_df_gauge_now)
del(concat_df_gauge_now)
print("Done")

#imerf final, imerg late, imerg early
track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_imerg_final/*.parquet"))
concat_df_imerg_final = dd.read_parquet(track_files).compute()
pps_convective_imerg_final, radar_convective_imerg_final, pps_nonconvective_imerg_final, radar_nonconvective_imerg_final, pps_all_imerg_final, radar_all_imerg_final = aplicator.concatenate_convec_and_nonconvec(concat_df_imerg_final)
array_x_convec_imerg_final, array_y_convec_imerg_final, array_x_nonconv_imerg_final, array_y_nonconv_imerg_final,array_x_all_imerg_final,array_y_all_imerg_final= aplicator.concatenate_array_positions(concat_df_imerg_final)
del(concat_df_imerg_final)
print("Done")

track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_imerg_late/*.parquet"))
concat_df_imerg_late = dd.read_parquet(track_files).compute()
pps_convective_imerg_late, radar_convective_imerg_late, pps_nonconvective_imerg_late, radar_nonconvective_imerg_late, pps_all_imerg_late, radar_all_imerg_late= aplicator.concatenate_convec_and_nonconvec(concat_df_imerg_late)
array_x_convec_imerg_late, array_y_convec_imerg_late, array_x_nonconv_imerg_late, array_y_nonconv_imerg_late,array_x_all_imerg_late,array_y_all_imerg_late= aplicator.concatenate_array_positions(concat_df_imerg_late)
del(concat_df_imerg_late)
print("Done")

track_files = sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_imerg_early/*.parquet"))
concat_df_imerg_early = dd.read_parquet(track_files).compute()
pps_convective_imerg_early, radar_convective_imerg_early, pps_nonconvective_imerg_early, radar_nonconvective_imerg_early, pps_all_imerg_early, radar_all_imerg_early= aplicator.concatenate_convec_and_nonconvec(concat_df_imerg_early)
array_x_convec_imerg_early, array_y_convec_imerg_early, array_x_nonconv_imerg_early, array_y_nonconv_imerg_early, array_x_all_imerg_early, array_y_all_imerg_early= aplicator.concatenate_array_positions(concat_df_imerg_early)
del(concat_df_imerg_early)
print("Done")

def find_unique_pairs(x, y, a):
    pairs = np.column_stack((x, y))
    unique_pairs, unique_indices, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros(unique_pairs.shape[0])
    np.add.at(sums, unique_indices, a)
    return unique_pairs, sums,counts

def map_values(array_x,array_y, pps,radar,shape,product):
    map_general=np.zeros((4,*shape))
    up_pps, sums_pps, counts_pps = find_unique_pairs(array_x, array_y, pps)
    up_radar, sums_radar, counts_radar = find_unique_pairs(array_x, array_y, radar)
    up_dif, sums_dif, counts_dif = find_unique_pairs(array_x, array_y, pps-radar)
    map_general[0,up_pps[:,0].astype(int),up_pps[:,1].astype(int)]=sums_pps
    map_general[1,up_pps[:,0].astype(int),up_pps[:,1].astype(int)]=sums_dif
    map_general[2,up_radar[:,0].astype(int),up_radar[:,1].astype(int)]=sums_radar
    map_general[3,up_radar[:,0].astype(int),up_radar[:,1].astype(int)]=counts_radar
    np.save(f"/home/milton/Documents/SBSR/Maps/{product}_map",map_general)

map_values(array_x_convec_gsmap_mvk, array_y_convec_gsmap_mvk, pps_convective_gsmap_mvk, radar_convective_gsmap_mvk,(232,291),"convec_gsmap_mvk")
map_values(array_x_nonconvec_gsmap_mvk, array_y_nonconvec_gsmap_mvk, pps_nonconvective_gsmap_mvk, radar_nonconvective_gsmap_mvk,(232,291),"nonconvec_gsmap_mvk")
map_values(array_x_all_gsmap_mvk, array_y_all_gsmap_mvk, pps_all_gsmap_mvk, radar_all_gsmap_mvk,(232,291),"all_gsmap_mvk")

map_values(array_x_convec_gsmap_nrt, array_y_convec_gsmap_nrt, pps_convective_gsmap_nrt, radar_convective_gsmap_nrt,(232,291),"convec_gsmap_nrt")
map_values(array_x_nonconvec_gsmap_nrt, array_y_nonconvec_gsmap_nrt, pps_nonconvective_gsmap_nrt, radar_nonconvective_gsmap_nrt,(232,291),"nonconvec_gsmap_nrt")
map_values(array_x_all_gsmap_nrt, array_y_all_gsmap_nrt, pps_all_gsmap_nrt, radar_all_gsmap_nrt,(232,291),"all_gsmap_nrt")

map_values(array_x_convec_gsmap_now, array_y_convec_gsmap_now, pps_convective_gsmap_now, radar_convective_gsmap_now,(232,291),"convec_gsmap_now")
map_values(array_x_nonconvec_gsmap_now, array_y_nonconvec_gsmap_now, pps_nonconvective_gsmap_now, radar_nonconvective_gsmap_now,(232,291),"nonconvec_gsmap_now")
map_values(array_x_all_gsmap_now, array_y_all_gsmap_now, pps_all_gsmap_now, radar_all_gsmap_now,(232,291),"all_gsmap_now")

map_values(array_x_convec_gauge_mvk, array_y_convec_gauge_mvk, pps_convective_gauge_mvk, radar_convective_gauge_mvk,(232,291),"convec_gauge_mvk")
map_values(array_x_nonconvec_gauge_mvk, array_y_nonconvec_gauge_mvk, pps_nonconvective_gauge_mvk, radar_nonconvective_gauge_mvk,(232,291),"nonconvec_gauge_mvk")
map_values(array_x_all_gauge_mvk, array_y_all_gauge_mvk, pps_all_gauge_mvk, radar_all_gauge_mvk,(232,291),"all_gauge_mvk")

map_values(array_x_convec_gauge_nrt, array_y_convec_gauge_nrt, pps_convective_gauge_nrt, radar_convective_gauge_nrt,(232,291),"convec_gauge_nrt")
map_values(array_x_nonconvec_gauge_nrt, array_y_nonconvec_gauge_nrt, pps_nonconvective_gauge_nrt, radar_nonconvective_gauge_nrt,(232,291),"nonconvec_gauge_nrt")
map_values(array_x_all_gauge_nrt, array_y_all_gauge_nrt, pps_all_gauge_nrt, radar_all_gauge_nrt,(232,291),"all_gauge_nrt")

map_values(array_x_convec_gauge_now, array_y_convec_gauge_now, pps_convective_gauge_now, radar_convective_gauge_now,(232,291),"convec_gauge_now")
map_values(array_x_nonconvec_gauge_now, array_y_nonconvec_gauge_now, pps_nonconvective_gauge_now, radar_nonconvective_gauge_now,(232,291),"nonconvec_gauge_now")
map_values(array_x_all_gauge_now, array_y_all_gauge_now, pps_all_gauge_now, radar_all_gauge_now,(232,291),"all_gauge_now")

#shape (231,291)
map_values(array_x_convec_imerg_final, array_y_convec_imerg_final, pps_convective_imerg_final, radar_convective_imerg_final,(231,291),"convec_imerg_final")
map_values(array_x_nonconv_imerg_final, array_y_nonconv_imerg_final, pps_nonconvective_imerg_final, radar_nonconvective_imerg_final,(231,291),"nonconvec_imerg_final")
map_values(array_x_all_imerg_final, array_y_all_imerg_final, pps_all_imerg_final, radar_all_imerg_final,(231,291),"all_imerg_final")

#shape (151,151)
map_values(array_x_convec_imerg_late, array_y_convec_imerg_late, pps_convective_imerg_late, radar_convective_imerg_late,(151,151),"convec_imerg_late")
map_values(array_x_nonconv_imerg_late, array_y_nonconv_imerg_late, pps_nonconvective_imerg_late, radar_nonconvective_imerg_late,(151,151),"nonconvec_imerg_late")
map_values(array_x_all_imerg_late, array_y_all_imerg_late, pps_all_imerg_late, radar_all_imerg_late,(151,151),"all_imerg_late")

#shape (151,151)
map_values(array_x_convec_imerg_early, array_y_convec_imerg_early, pps_convective_imerg_early, radar_convective_imerg_early,(151,151),"convec_imerg_early")
map_values(array_x_nonconv_imerg_early, array_y_nonconv_imerg_early, pps_nonconvective_imerg_early, radar_nonconvective_imerg_early,(151,151),"nonconvec_imerg_early")
map_values(array_x_all_imerg_early, array_y_all_imerg_early, pps_all_imerg_early, radar_all_imerg_early,(151,151),"all_imerg_early")