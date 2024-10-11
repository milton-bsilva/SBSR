import numpy as np
import dask.dataframe as dd
import utils_validation
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import xarray as xr

def set_cartopy(ax,gll):
    ax.set_extent([-56.25, -52.5, -22, -18.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=.6)
    ax.add_feature(cfeature.STATES, linewidth=.3)
    
    # Configure gridlines and labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=.2, color='gray', alpha=0.5, linestyle='--')
    
    # Show labels only on the bottom and left
    if gll=='so':
        pass
    if gll=='so_lateral':
        gl.left_labels = True
        gl.bottom_labels = False
        gl.right_labels = False
        gl.top_labels = False
        #set yticks
        gl.ylocator = mticker.FixedLocator([-22, -21, -20, -19,])
    if gll=='so_fundo':
        gl.left_labels = False
        gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
    if gll=='canto':
        gl.left_labels = True
        gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
        gl.ylocator = mticker.FixedLocator([-22, -21, -20, -19,])


    

def load_np(path,invert=True):
    data = np.load(path)
    data[data == 0] = np.nan
    data2=np.full((data.shape),np.nan)
    if invert:
        for i in range(len(data[0])):
            for j in range(len(data[1])):
                try:
                    data2[:,i,j]=data[:,j,i]
                except:
                    pass
    else:
        data2=data.copy()
    return data2

#load maps again
convec_gsmap_mvk=load_np("/home/milton/Documentos/SBSR/Maps/convec_gsmap_mvk_map.npy")
nonconvec_gsmap_mvk=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_gsmap_mvk_map.npy")
all_gsmap_mvk=load_np("/home/milton/Documentos/SBSR/Maps/all_gsmap_mvk_map.npy")

convec_gsmap_nrt=load_np("/home/milton/Documentos/SBSR/Maps/convec_gsmap_nrt_map.npy")
nonconvec_gsmap_nrt=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_gsmap_nrt_map.npy")
all_gsmap_nrt=load_np("/home/milton/Documentos/SBSR/Maps/all_gsmap_nrt_map.npy",invert=False)

convec_gsmap_now=load_np("/home/milton/Documentos/SBSR/Maps/convec_gsmap_now_map.npy")
nonconvec_gsmap_now=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_gsmap_now_map.npy")
all_gsmap_now=load_np("/home/milton/Documentos/SBSR/Maps/all_gsmap_now_map.npy")

convec_gauge_mvk=load_np("/home/milton/Documentos/SBSR/Maps/convec_gauge_mvk_map.npy")
nonconvec_gauge_mvk=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_gauge_mvk_map.npy")
all_gauge_mvk=load_np("/home/milton/Documentos/SBSR/Maps/all_gauge_mvk_map.npy",invert=False)

convec_gauge_nrt=load_np("/home/milton/Documentos/SBSR/Maps/convec_gauge_nrt_map.npy")
nonconvec_gauge_nrt=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_gauge_nrt_map.npy")
all_gauge_nrt=load_np("/home/milton/Documentos/SBSR/Maps/all_gauge_nrt_map.npy",invert=False)

convec_gauge_now=load_np("/home/milton/Documentos/SBSR/Maps/convec_gauge_now_map.npy")
nonconvec_gauge_now=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_gauge_now_map.npy")
all_gauge_now=load_np("/home/milton/Documentos/SBSR/Maps/all_gauge_now_map.npy")

convec_imerg_final=load_np("/home/milton/Documentos/SBSR/Maps/convec_imerg_final_map.npy")
nonconvec_imerg_final=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_imerg_final_map.npy")
all_imerg_final=load_np("/home/milton/Documentos/SBSR/Maps/all_imerg_final_map.npy")

convec_imerg_late=load_np("/home/milton/Documentos/SBSR/Maps/convec_imerg_late_map.npy")
nonconvec_imerg_late=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_imerg_late_map.npy")
all_imerg_late=load_np("/home/milton/Documentos/SBSR/Maps/all_imerg_late_map.npy")

convec_imerg_early=load_np("/home/milton/Documentos/SBSR/Maps/convec_imerg_early_map.npy")
nonconvec_imerg_early=load_np("/home/milton/Documentos/SBSR/Maps/nonconvec_imerg_early_map.npy")
all_imerg_early=load_np("/home/milton/Documentos/SBSR/Maps/all_imerg_early_map.npy")

Lon=np.round(np.append(np.arange(0,180,.1),np.arange(-180,0,0.1)),2)
Lat=np.round(np.arange(60,-60,-.1),2)
X_grid_gsmap,Y_grid_gsmap=np.meshgrid(Lon[2936:3227],Lat[686:918])    

dfinal=xr.open_dataset("/home/milton/Documentos/SBSR/3B-HHR.MS.MRG.3IMERG.20180101-S000000.HDF5.nc4")
X_final,Y_final=np.meshgrid(dfinal.lon.values,dfinal.lat.values)

dimerg=xr.open_dataset("/home/milton/Documentos/SBSR/3B-HHR-L.MS.MRG.3IMERG.20180101-S000000.HDF5.nc4")
X_imerg,Y_imerg=np.meshgrid(dimerg.lon.values,dimerg.lat.values)
circ=np.loadtxt('/home/milton/Downloads/circ.txt')


def contornos(axs, data, X, Y,colorbar=False):
    data_plot=data[1]/data[3]
    cr=axs.contour(X,Y,data_plot, colors='k', transform=ccrs.PlateCarree(), levels=np.arange(0,2.6,.1),linewidths=0.5)
    cf=axs.contourf(X,Y,data_plot, cmap='winter_r', transform=ccrs.PlateCarree(), levels=np.arange(0,2.6,.1),extend='both')
    if colorbar:
        cbar=plt.colorbar(cf, ax=ax, orientation='vertical', aspect=50, shrink=0.8)
        cbar.set_label(r'mm h$^{-1}$')


fig, ax = plt.subplots(9, 3, figsize=(8, 20), subplot_kw={'projection': ccrs.PlateCarree()})
ax=ax.ravel()
for i in range(27):
    if i==24:
        set_cartopy(ax[i],'canto')
    elif i==0 or i%3==0:
        set_cartopy(ax[i],'so_lateral')
    elif i==25 or i==26:
        set_cartopy(ax[i],'so_fundo')
    else:
        set_cartopy(ax[i],'so')
    

contornos(ax[0],all_imerg_final,X_final,Y_final)
contornos(ax[1],convec_imerg_final,X_final,Y_final)
contornos(ax[2],nonconvec_imerg_final,X_final,Y_final)

contornos(ax[3],all_imerg_late,X_imerg,Y_imerg)
contornos(ax[4],convec_imerg_late,X_imerg,Y_imerg)
contornos(ax[5],nonconvec_imerg_late,X_imerg,Y_imerg)

contornos(ax[6],all_imerg_early,X_imerg,Y_imerg)
contornos(ax[7],convec_imerg_early,X_imerg,Y_imerg)
contornos(ax[8],nonconvec_imerg_early,X_imerg,Y_imerg)

contornos(ax[9],all_gsmap_mvk,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[10],convec_gsmap_mvk,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[11],nonconvec_gsmap_mvk,X_grid_gsmap,Y_grid_gsmap)

contornos(ax[12],all_gsmap_nrt,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[13],convec_gsmap_nrt,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[14],nonconvec_gsmap_nrt,X_grid_gsmap,Y_grid_gsmap)

contornos(ax[15],all_gsmap_now,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[16],convec_gsmap_now,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[17],nonconvec_gsmap_now,X_grid_gsmap,Y_grid_gsmap)

contornos(ax[18],all_gauge_mvk,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[19],convec_gauge_mvk,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[20],nonconvec_gauge_mvk,X_grid_gsmap,Y_grid_gsmap)

contornos(ax[21],all_gauge_nrt,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[22],convec_gauge_nrt,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[23],nonconvec_gauge_nrt,X_grid_gsmap,Y_grid_gsmap)

contornos(ax[24],all_gauge_now,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[25],convec_gauge_now,X_grid_gsmap,Y_grid_gsmap)
contornos(ax[26],nonconvec_gauge_now,X_grid_gsmap,Y_grid_gsmap)
plt.tight_layout()
plt.savefig('/home/milton/Documentos/SBSR/Figuras/Mapas.png',dpi=300,bbox_inches='tight')

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

new_order=[4,5,3,6,8,7,0,2,1]
colors=['orange','orange','orange','green','green','green','pink','pink','pink']
markers = ['s', '^', 'o', 's', '^', 'o', 's', '^', 'o']
linestyles=['--','-.',':','--','-.',':','--','-.',':',]
products=['IMERG-F', 'IMERG-L', 'IMERG-E', 'GSMaP-MVK', 'GSMaP-NRT', 'GSMaP-Now', 'GSMaP-MVK-G', 'GSMaP-NRT-G', 'GSMaP-Now-G']
metrics_in_portuguese = [r'Erro Absoluto Médio (mm h$^{-1}$)', r'Raiz do Erro Quadrático Médio (mm h$^{-1}$)',r'Viés (mm h$^{-1}$)','Correlação de Spearman']


fig, ax = plt.subplots(4, 3, figsize=(10, 18))
fl=sorted(glob('/home/milton/Documentos/SBSR/percentiles/*all_percentile.csv'))
fl=[fl[i] for i in new_order]
for j,files in enumerate(fl):
    df=pd.read_csv(files)
    ax[0,0].plot(df.iloc[0,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                 lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[1,0].plot(df.iloc[1,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[2,0].plot(df.iloc[2,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[3,0].plot(df.iloc[3,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
                 

fl=sorted(glob('/home/milton/Documentos/SBSR/percentiles/*_convective_percentile.csv'))
fl=[fl[i] for i in new_order]

for j,files in enumerate(fl):
    df=pd.read_csv(files)
    ax[0,1].plot(df.iloc[0,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                 lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[1,1].plot(df.iloc[1,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[2,1].plot(df.iloc[2,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[3,1].plot(df.iloc[3,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)



fl=sorted(glob('/home/milton/Documentos/SBSR/percentiles/*_nonconvective_percentile.csv'))
fl=[fl[i] for i in new_order]
for j,files in enumerate(fl):
    df=pd.read_csv(files)
    ax[0,2].plot(df.iloc[0,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                 lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7,label=products[j])
    ax[1,2].plot(df.iloc[1,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[2,2].plot(df.iloc[2,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[3,2].plot(df.iloc[3,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)

ax[0,2].legend()
ylim_values = ([0, 7.5], [0, 10.75], [-.250, 7.25], [0, .425])
yticks_values=([1,2,3,4,5,6,7], [2,4,6,8,10], [1,2,3,4,5,6,7], [.1,.2,.3,.4])
for i in range(4):
    for j in range(3):
        ax[i, j].set_ylim(ylim_values[i])
        ax[i, j].set_yticks(yticks_values[i])
        if j != 0:
            ax[i, j].set_yticks(yticks_values[i],['']*len(yticks_values[i]))
        else:
            ax[i, j].set_ylabel(metrics_in_portuguese[i])

        if i == 3:
            ax[i, j].set_xticks([0,1,2],['< 33º', '33º - 66º', '> 66º'],rotation=45) 
        else:
            ax[i, j].set_xticks([])
        ax[i, j].yaxis.grid(True)

plt.tight_layout()
plt.savefig('/home/milton/Documentos/SBSR/Figuras/Percentiles.png',dpi=300,bbox_inches='tight')