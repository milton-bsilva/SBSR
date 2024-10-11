import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

new_order=[4,5,3,6,8,7,0,2,1]
colors=['orange','orange','orange','green','green','green','pink','pink','pink']
markers = ['s', '^', 'o', 's', '^', 'o', 's', '^', 'o']
linestyles=['--','-.',':','--','-.',':','--','-.',':',]
products=['IMERG-F', 'IMERG-L', 'IMERG-E', 'GSMaP-MVK', 'GSMaP-NRT', 'GSMaP-Now', 'GSMaP-MVK-G', 'GSMaP-NRT-G', 'GSMaP-Now-G']
metrics_in_portuguese = [r'Erro Absoluto Médio (mm h$^{-1}$)', r'Raiz do Erro Quadrático Médio (mm h$^{-1}$)',r'Viés (mm h$^{-1}$)','Correlação de Spearman']


fig, ax = plt.subplots(3, 3, figsize=(10, 10))
fl=sorted(glob('/home/milton/Documents/SBSR/percentiles/*all_percentile.csv'))
fl=[fl[i] for i in new_order]
for j,files in enumerate(fl):
    df=pd.read_csv(files)
    ax[0,0].plot(df.iloc[0,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                 lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[1,0].plot(df.iloc[1,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[2,0].plot(df.iloc[2,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    #ax[3,0].plot(df.iloc[3,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
    #                lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
                 

fl=sorted(glob('/home/milton/Documents/SBSR/percentiles/*_convective_percentile.csv'))
fl=[fl[i] for i in new_order]

for j,files in enumerate(fl):
    df=pd.read_csv(files)
    ax[0,1].plot(df.iloc[0,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                 lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[1,1].plot(df.iloc[1,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[2,1].plot(df.iloc[2,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    #ax[3,1].plot(df.iloc[3,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
    #                lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)



fl=sorted(glob('/home/milton/Documents/SBSR/percentiles/*_nonconvective_percentile.csv'))
fl=[fl[i] for i in new_order]
for j,files in enumerate(fl):
    df=pd.read_csv(files)
    ax[0,2].plot(df.iloc[0,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                 lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7,label=products[j])
    ax[1,2].plot(df.iloc[1,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    ax[2,2].plot(df.iloc[2,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
                    lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)
    #ax[3,2].plot(df.iloc[3,1:].values,c=colors[j],marker=markers[j], markersize=10, markerfacecolor=colors[j],
    #                lw=3,linestyle=linestyles[j], markeredgecolor='k',alpha=.7)

ax[0,2].legend(loc=2)
ylim_values = ([0, 7.5], [0, 10.75], [-.250, 7.25], [0, .425])
yticks_values=([1,2,3,4,5,6,7], [2,4,6,8,10], [1,2,3,4,5,6,7], [.1,.2,.3,.4])
for i in range(3):
    for j in range(3):
        ax[i, j].set_ylim(ylim_values[i])
        ax[i, j].set_yticks(yticks_values[i])
        if j != 0:
            ax[i, j].set_yticks(yticks_values[i],['']*len(yticks_values[i]))
        else:
            ax[i, j].set_ylabel(metrics_in_portuguese[i])

        if i == 2:
            ax[i, j].set_xticks([0,1,2],['< 33º', '33º - 66º', '> 66º']) 
        else:
            ax[i, j].set_xticks([])
        ax[i, j].yaxis.grid(True)


ax[0,0].set_title('Precipitação total')
ax[0,1].set_title('Precipitação convectiva')
ax[0,2].set_title('Precipitação não-convectiva')

plt.tight_layout()
plt.savefig('/home/milton/Documents/SBSR/Figuras/Percentiles_wosrc.png',dpi=300,bbox_inches='tight')