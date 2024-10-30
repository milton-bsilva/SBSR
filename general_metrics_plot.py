import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def set_boxplot(ax, df, data,positions):
    col = df[f'all_{data}']
    i = positions[0]
    ax.add_patch(plt.Rectangle((i, col[1]), 0.18, col[3] - col[1], fill=True, color='.5'))
    ax.plot([i, i + 0.18], [col[2], col[2]], color='black', lw=2)
    ax.plot([i + 0.09, i + 0.09], [col[0], col[4]], color='black')
    col = df[f'convective_{data}']
    i = positions[1]
    ax.add_patch(plt.Rectangle((i, col[1]), 0.18, col[3] - col[1], fill=True, color='r'))
    ax.plot([i, i + 0.18], [col[2], col[2]], color='black', lw=2)
    ax.plot([i + 0.09, i + 0.09], [col[0], col[4]], color='black')
    col = df[f'stratiform_{data}']
    i = positions[2]
    ax.add_patch(plt.Rectangle((i, col[1]), 0.18, col[3] - col[1], fill=True, color='b'))
    ax.plot([i, i + 0.18], [col[2], col[2]], color='black', lw=2)
    ax.plot([i + 0.09, i + 0.09], [col[0], col[4]], color='black')


general_metrics_data = {'metrics': ['MAE', 'RMSE', 'Bias', 'Spearman Correlation', 'Spearman Correlation p value'],
    'convective': [2.938662, 6.422709, 1.629815, 2.459961, 2.389446],
    'stratiform': [1.473896, 1.687408, 8.426250, 9.916509, 0.311908],
    'all': [1.860545, 7.941882, 5.140811, 4.803117, 3.656702],
    'boxplot': ['lower_fence', 'q1', 'median', 'q3', 'upper_fence'],
    'convective_boxplot': [0.795734, 0.674506, 4.674828, 4.670422, 7.282694],
    'stratiform_boxplot': [8.490597, 2.850098, 5.757925, 8.270954, 1.369723],
    'all_boxplot': [8.304391, 5.182959, 0.562455, 7.454321, 8.701157]}

phase='begin'
general_final=pd.read_csv(f'/media/milton/Data/Data/Validation/track_imerg_final/metrics_{phase}.csv')
general_late=pd.read_csv(f'/media/milton/Data/Data/Validation/track_imerg_latest/metrics_{phase}.csv')
general_early=pd.read_csv(f'/media/milton/Data/Data/Validation/track_imerg_early/metrics_{phase}.csv')

#gsmap
general_mvk=pd.read_csv(f'/media/milton/Data/Data/Validation/track_gsmap_mvk/metrics_{phase}.csv')
general_nrt=pd.read_csv(f'/media/milton/Data/Data/Validation/track_gsmap_nrt/metrics_{phase}.csv')
general_now=pd.read_csv(f'/media/milton/Data/Data/Validation/track_gsmap_now/metrics_{phase}.csv')

general_gauge_mvk=pd.read_csv(f'/media/milton/Data/Data/Validation/track_gauge_mvk/metrics_{phase}.csv')
general_gauge_nrt=pd.read_csv(f'/media/milton/Data/Data/Validation/track_gauge_nrt/metrics_{phase}.csv')
general_gauge_now=pd.read_csv(f'/media/milton/Data/Data/Validation/track_gauge_now/metrics_{phase}.csv')
general_merge=pd.read_csv(f'/media/milton/Data/Data/Validation/track_merge/metrics_{phase}.csv')

#copy general metrics data to all df above
general_final=general_metrics_data.copy()
general_late=general_metrics_data.copy()
general_early=general_metrics_data.copy()

general_mvk=general_metrics_data.copy()
general_nrt=general_metrics_data.copy()
general_now=general_metrics_data.copy()

general_gauge_mvk=general_metrics_data.copy()
general_gauge_nrt=general_metrics_data.copy()
general_gauge_now=general_metrics_data.copy()

general_merge=general_metrics_data.copy()

sufix_products = ['imerg_final','imerg_late','imerg_early','mvk','nrt','now','gauge_mvk','gauge_nrt','gauge_now']
metrics_in_portuguese = ['Erro Absoluto Médio', 'Raiz do Erro Quadrático Médio','Viés','Correlação de Spearman']
indexe=np.array([0,.8,1.6,2.4,3.2,4,4.8,5.6,6.4,7.2])
bar_width = 0.2
offset = bar_width

fig, ax = plt.subplots(5, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1, 1]})
ax=ax.ravel()
axb=ax[0]
set_boxplot(axb,general_final,'pps',[0,.2,.4])
set_boxplot(axb,general_late,'pps',[.8,1,1.2])
set_boxplot(axb,general_early,'pps',[1.6,1.8,2])
set_boxplot(axb,general_mvk,'pps',[2.4,2.6,2.8])
set_boxplot(axb,general_nrt,'pps',[3.2,3.4,3.6])
set_boxplot(axb,general_now,'pps',[4,4.2,4.4])
set_boxplot(axb,general_gauge_mvk,'pps',[4.8,5,5.2])
set_boxplot(axb,general_gauge_nrt,'pps',[5.6,5.8,6])
set_boxplot(axb,general_gauge_now,'pps',[6.4,6.6,6.8])
set_boxplot(axb,general_merge,'pps',[7.2,7.4,7.6])

axb.set_title('Medições de Satélite')
axb.set_ylabel(r'mm h$^{-1}$')
axb.set_xticks([])
axb.set_xticklabels([])
axb.set_xlim(-.2,7.8)
axb.set_ylim(0,11)
axb.yticks(np.arange(0,12,2))
axb.grid(axis='y')
axb.set_xticks(indexe+offset)
axb.set_xticklabels(['', '', '', '', '', '', '', '', ''])

axc=ax[1]
set_boxplot(axc,general_final,'radar',[0,.2,.4])
set_boxplot(axc,general_late,'radar',[.8,1,1.2])
set_boxplot(axc,general_early,'radar',[1.6,1.8,2])
set_boxplot(axc,general_mvk,'radar',[2.4,2.6,2.8])
set_boxplot(axc,general_nrt,'radar',[3.2,3.4,3.6])
set_boxplot(axc,general_now,'radar',[4,4.2,4.4])
set_boxplot(axc,general_gauge_mvk,'radar',[4.8,5,5.2])
set_boxplot(axc,general_gauge_nrt,'radar',[5.6,5.8,6])
set_boxplot(axc,general_gauge_now,'radar',[6.4,6.6,6.8])
set_boxplot(axc,general_merge,'radar',[7.2,7.4,7.6])

axc.set_ylim(0,16)
axc.set_title('Medições de Radar')
axc.set_ylabel(r'mm h$^{-1}$')
axc.set_xticks([])
axc.set_xticklabels([])
axc.set_xlim(-.2,7.8)
axc.set_ylim(0,11)
axc.yticks(np.arange(0,12,2))
axc.grid(axis='y')
axc.set_xticks(indexe+offset)
axc.set_xticklabels(['', '', '', '', '', '', '', '', ''])


axs = ax[2:]

for i in range(3):
    axm = axs[i]  # Get the current subplot
    for j, dataset in enumerate(sufix_products):
        axm.bar(indexe[j], general_metrics_df[f'all_{dataset}'].iloc[i], color='.5', width=bar_width,edgecolor='black')
        axm.bar(indexe[j]+offset, general_metrics_df[f'convective_{dataset}'].iloc[i], color='r', width=bar_width,edgecolor='black')
        axm.bar(indexe[j]+2*offset, general_metrics_df[f'nonconvective_{dataset}'].iloc[i], color='b', width=bar_width,edgecolor='black')
        axm.set_title(metrics_in_portuguese[i])
        axm.set_xticks([])
        axm.set_xticklabels([])
        axm.set_ylabel(r'mm h$^{-1}$')
        axm.set_xlim(-.2,7)
        axm.set_xticks(indexe+offset)
        axm.set_xticklabels(['', '', '', '', '', '', '', '', ''])

axm.set_ylabel('')
axm.set_xticks(indexe+offset)
#plot dashed line in y=0
axm.hlines(0,-2,7,linestyles='dashed',linewidth=.25)
axm.set_xticklabels(['IMERG-F', 'IMERG-L', 'IMERG-E', 'GSMaP-MVK', 'GSMaP-NRT', 'GSMaP-Now', 'GSMaP-MVK-G', 'GSMaP-NRT-G', 'GSMaP-Now-G'],fontsize=8)   




