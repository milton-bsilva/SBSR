import numpy as np
import dask.dataframe as dd
import utils_validation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

calculator = utils_validation.MetricsCalculator()
aplicator = utils_validation.MetricsApplicator(calculator) 


from datetime import datetime

def get_date(track_files):
    file_names = [f.split('/')[-1].replace('.parquet', '') for f in track_files]
    start_date = datetime(2019, 7, 1)
    dates = [datetime.strptime(f, "%Y%m%d_%H%M") for f in file_names]
    index = next(i for i, date in enumerate(dates) if date >= start_date)
    return track_files[index:]

track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_mvk/*.parquet")))
concat_df_gsmap_mvk = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_gsmap_mvk, radar_convective_gsmap_mvk, pps_nonconvective_gsmap_mvk, radar_nonconvective_gsmap_mvk, pps_all_gsmap_mvk, radar_all_gsmap_mvk= aplicator.concatenate_convec_and_nonconvec(concat_df_gsmap_mvk)
del(concat_df_gsmap_mvk)

print('ok')
#gsmap_nrt
track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_nrt/*.parquet")))
concat_df_gsmap_nrt = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_gsmap_nrt, radar_convective_gsmap_nrt, pps_nonconvective_gsmap_nrt, radar_nonconvective_gsmap_nrt, pps_all_gsmap_nrt, radar_all_gsmap_nrt= aplicator.concatenate_convec_and_nonconvec(concat_df_gsmap_nrt)
del(concat_df_gsmap_nrt)

print('ok')
#gsmap_now
track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_now/*.parquet")))
concat_df_gsmap_now = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_gsmap_now, radar_convective_gsmap_now, pps_nonconvective_gsmap_now, radar_nonconvective_gsmap_now, pps_all_gsmap_now, radar_all_gsmap_now= aplicator.concatenate_convec_and_nonconvec(concat_df_gsmap_now)
del(concat_df_gsmap_now)

print('ok')
#gauge mvk, gauge nrt, gauge now
track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_gauge_mvk/*.parquet")))
concat_df_gauge_mvk = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_gauge_mvk, radar_convective_gauge_mvk, pps_nonconvective_gauge_mvk, radar_nonconvective_gauge_mvk, pps_all_gauge_mvk, radar_all_gauge_mvk= aplicator.concatenate_convec_and_nonconvec(concat_df_gauge_mvk)
del(concat_df_gauge_mvk)

print('ok')
track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_gauge_nrt/*.parquet")))
concat_df_gauge_nrt = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_gauge_nrt, radar_convective_gauge_nrt, pps_nonconvective_gauge_nrt, radar_nonconvective_gauge_nrt,pps_all_gauge_nrt, radar_all_gauge_nrt= aplicator.concatenate_convec_and_nonconvec(concat_df_gauge_nrt)
del(concat_df_gauge_nrt)

print('ok')
track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_gsmap_gauge_now/*.parquet")))
concat_df_gauge_now = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_gauge_now, radar_convective_gauge_now, pps_nonconvective_gauge_now, radar_nonconvective_gauge_now, pps_all_gauge_now, radar_all_gauge_now= aplicator.concatenate_convec_and_nonconvec(concat_df_gauge_now)
del(concat_df_gauge_now)
print('ok')

#imerf final, imerg late, imerg early
track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_imerg_final/*.parquet")))
concat_df_imerg_final = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_imerg_final, radar_convective_imerg_final, pps_nonconvective_imerg_final, radar_nonconvective_imerg_final, pps_all_imerg_final, radar_all_imerg_final = aplicator.concatenate_convec_and_nonconvec(concat_df_imerg_final)
del(concat_df_imerg_final)
print('ok')

track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_imerg_late/*.parquet")))
concat_df_imerg_late = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_imerg_late, radar_convective_imerg_late, pps_nonconvective_imerg_late, radar_nonconvective_imerg_late, pps_all_imerg_late, radar_all_imerg_late= aplicator.concatenate_convec_and_nonconvec(concat_df_imerg_late)
del(concat_df_imerg_late)
print('ok')

track_files = get_date(sorted(glob(f"/media/milton/Data/Data/Tables_optim/track_imerg_early/*.parquet")))
concat_df_imerg_early = dd.read_parquet(track_files,columns=['radar_values','array_values','classification']).compute()
pps_convective_imerg_early, radar_convective_imerg_early, pps_nonconvective_imerg_early, radar_nonconvective_imerg_early, pps_all_imerg_early, radar_all_imerg_early= aplicator.concatenate_convec_and_nonconvec(concat_df_imerg_early)
del(concat_df_imerg_early)
print('ok')

general_metrics_df = pd.DataFrame({'metrics': ['MAE', 'RMSE', 'Bias', 'Spearman Correlation', 
    'Spearman Correlation p value'],
    'convective_imerg_final': aplicator.calculate_array_metrics(pps_convective_imerg_final, radar_convective_imerg_final),
    'nonconvective_imerg_final': aplicator.calculate_array_metrics(pps_nonconvective_imerg_final, radar_nonconvective_imerg_final),
    'all_imerg_final': aplicator.calculate_array_metrics(pps_all_imerg_final, radar_all_imerg_final),
    'convective_imerg_late': aplicator.calculate_array_metrics(pps_convective_imerg_late, radar_convective_imerg_late),
    'nonconvective_imerg_late': aplicator.calculate_array_metrics(pps_nonconvective_imerg_late, radar_nonconvective_imerg_late),
    'all_imerg_late': aplicator.calculate_array_metrics(pps_all_imerg_late, radar_all_imerg_late),
    'convective_imerg_early': aplicator.calculate_array_metrics(pps_convective_imerg_early, radar_convective_imerg_early),
    'nonconvective_imerg_early': aplicator.calculate_array_metrics(pps_nonconvective_imerg_early, radar_nonconvective_imerg_early),
    'all_imerg_early': aplicator.calculate_array_metrics(pps_all_imerg_early, radar_all_imerg_early),
    'convective_mvk': aplicator.calculate_array_metrics(pps_convective_gsmap_mvk, radar_convective_gsmap_mvk),
    'nonconvective_mvk': aplicator.calculate_array_metrics(pps_nonconvective_gsmap_mvk, radar_nonconvective_gsmap_mvk),
    'all_mvk': aplicator.calculate_array_metrics(pps_all_gsmap_mvk, radar_all_gsmap_mvk),
    'convective_nrt': aplicator.calculate_array_metrics(pps_convective_gsmap_nrt, radar_convective_gsmap_nrt),
    'nonconvective_nrt': aplicator.calculate_array_metrics(pps_nonconvective_gsmap_nrt, radar_nonconvective_gsmap_nrt),
    'all_nrt': aplicator.calculate_array_metrics(pps_all_gsmap_nrt, radar_all_gsmap_nrt),
    'convective_now': aplicator.calculate_array_metrics(pps_convective_gsmap_now, radar_convective_gsmap_now),
    'nonconvective_now': aplicator.calculate_array_metrics(pps_nonconvective_gsmap_now, radar_nonconvective_gsmap_now),
    'all_now': aplicator.calculate_array_metrics(pps_all_gsmap_now, radar_all_gsmap_now),
    'convective_gauge_mvk': aplicator.calculate_array_metrics(pps_convective_gauge_mvk, radar_convective_gauge_mvk),
    'nonconvective_gauge_mvk': aplicator.calculate_array_metrics(pps_nonconvective_gauge_mvk, radar_nonconvective_gauge_mvk),
    'all_gauge_mvk': aplicator.calculate_array_metrics(pps_all_gauge_mvk, radar_all_gauge_mvk),
    'convective_gauge_nrt': aplicator.calculate_array_metrics(pps_convective_gauge_nrt, radar_convective_gauge_nrt),
    'nonconvective_gauge_nrt': aplicator.calculate_array_metrics(pps_nonconvective_gauge_nrt, radar_nonconvective_gauge_nrt),
    'all_gauge_nrt': aplicator.calculate_array_metrics(pps_all_gauge_nrt, radar_all_gauge_nrt),
    'convective_gauge_now': aplicator.calculate_array_metrics(pps_convective_gauge_now, radar_convective_gauge_now),
    'nonconvective_gauge_now': aplicator.calculate_array_metrics(pps_nonconvective_gauge_now, radar_nonconvective_gauge_now),
    'all_gauge_now': aplicator.calculate_array_metrics(pps_all_gauge_now, radar_all_gauge_now)})

general_metrics_df.to_csv('/home/milton/Documents/SBSR/general_metrics.csv',index=False)

def setboxplot(axb, data1,data2,data3,positions,label=False):
    if label:
        sns.boxplot(data1,ax=axb,width=0.2,boxprops=dict(alpha=1), color='.5',positions=[positions[0]],label='Precipitatação total', showfliers=False)
        sns.boxplot(data2,ax=axb,width=0.2,boxprops=dict(alpha=1), color='r',positions=[positions[1]],label='Precipitação convectiva', showfliers=False)
        sns.boxplot(data3,ax=axb,width=0.2,boxprops=dict(alpha=1), color='b',positions=[positions[2]],label='Precipitação não-convectiva', showfliers=False)
        axb.legend()
    else:
        sns.boxplot(data1,ax=axb,width=0.2,boxprops=dict(alpha=1), color='.5',positions=[positions[0]], showfliers=False)
        sns.boxplot(data2,ax=axb,width=0.2,boxprops=dict(alpha=1), color='r',positions=[positions[1]], showfliers=False)
        sns.boxplot(data3,ax=axb,width=0.2,boxprops=dict(alpha=1), color='b',positions=[positions[2]], showfliers=False)


##########figura
sufix_products = ['imerg_final','imerg_late','imerg_early','mvk','nrt','now','gauge_mvk','gauge_nrt','gauge_now']
metrics_in_portuguese = ['Erro Absoluto Médio', 'Raiz do Erro Quadrático Médio','Viés','Correlação de Spearman']
indexe=np.array([0,.8,1.6,2.4,3.2,4,4.8,5.6,6.4])
bar_width = 0.2
offset = bar_width

fig, ax = plt.subplots(5, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1.75, 1.75, 1, 1, 1]})
ax=ax.ravel()
axb=ax[0]
setboxplot(axb,pps_all_imerg_final,pps_convective_imerg_final,pps_nonconvective_imerg_final,[0,.2,.4])
setboxplot(axb,pps_all_imerg_late,pps_convective_imerg_late,pps_nonconvective_imerg_late,[.8,1,1.2])
setboxplot(axb,pps_all_imerg_early,pps_convective_imerg_early,pps_nonconvective_imerg_early,[1.6,1.8,2])
setboxplot(axb,pps_all_gsmap_mvk,pps_convective_gsmap_mvk,pps_nonconvective_gsmap_mvk,[2.4,2.6,2.8])
setboxplot(axb,pps_all_gsmap_nrt,pps_convective_gsmap_nrt,pps_nonconvective_gsmap_nrt,[3.2,3.4,3.6])
setboxplot(axb,pps_all_gsmap_now,pps_convective_gsmap_now,pps_nonconvective_gsmap_now,[4,4.2,4.4])
setboxplot(axb,pps_all_gauge_mvk,pps_convective_gauge_mvk,pps_nonconvective_gauge_mvk,[4.8,5,5.2])
setboxplot(axb,pps_all_gauge_nrt,pps_convective_gauge_nrt,pps_nonconvective_gauge_nrt,[5.6,5.8,6])
setboxplot(axb,pps_all_gauge_now,pps_convective_gauge_now,pps_nonconvective_gauge_now,[6.4,6.6,6.8])
print('ok')

axb.set_title('Medições de Satélite')
axb.set_ylabel(r'mm h$^{-1}$')
axb.set_xticks([])
axb.set_xticklabels([])
axb.set_xlim(-.2,7)
axb.set_ylim(0,11)
axb.yticks(np.arange(0,12,2))
axb.grid(axis='y')
axb.set_xticks(indexe+offset)
axb.set_xticklabels(['', '', '', '', '', '', '', '', ''])

########
axc=ax[1]
setboxplot(axc,radar_all_imerg_final,radar_convective_imerg_final,radar_nonconvective_imerg_final,[0,.2,.4])
setboxplot(axc,radar_all_imerg_late,radar_convective_imerg_late,radar_nonconvective_imerg_late,[.8,1,1.2])
setboxplot(axc,radar_all_imerg_early,radar_convective_imerg_early,radar_nonconvective_imerg_early,[1.6,1.8,2])
setboxplot(axc,radar_all_gsmap_mvk,radar_convective_gsmap_mvk,radar_nonconvective_gsmap_mvk,[2.4,2.6,2.8])
setboxplot(axc,radar_all_gsmap_nrt,radar_convective_gsmap_nrt,radar_nonconvective_gsmap_nrt,[3.2,3.4,3.6])
setboxplot(axc,radar_all_gsmap_now,radar_convective_gsmap_now,radar_nonconvective_gsmap_now,[4,4.2,4.4])
setboxplot(axc,radar_all_gauge_mvk,radar_convective_gauge_mvk,radar_nonconvective_gauge_mvk,[4.8,5,5.2])
setboxplot(axc,radar_all_gauge_nrt,radar_convective_gauge_nrt,radar_nonconvective_gauge_nrt,[5.6,5.8,6])
setboxplot(axc,radar_all_gauge_now,radar_convective_gauge_now,radar_nonconvective_gauge_now,[6.4,6.6,6.8],label=True)
axc.set_ylim(0,16)
axc.set_title('Medições de Radar')
axc.set_ylabel(r'mm h$^{-1}$')
axc.set_xticks([])
axc.set_xticklabels([])
axc.set_xlim(-.2,7)
axc.set_ylim(0,11)
axc.yticks(np.arange(0,12,2))
axc.grid(axis='y')
axc.set_xticks(indexe+offset)
axc.set_xticklabels(['', '', '', '', '', '', '', '', ''])

print('ok')

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
plt.savefig('/home/milton/Documents/SBSR/Figuras/general_metrics_wosrc.png',bbox_inches='tight')
plt.close()
'''

def boxplot_stats(data):
    # Calcula os quartis e os valores mínimos e máximos
    quartil_inferior = np.percentile(data, 25)  # Primeiro quartil (Q1)
    mediana = np.median(data)  # Mediana (Q2)
    quartil_superior = np.percentile(data, 75)  # Terceiro quartil (Q3)
    
    # Limites do boxplot (mínimo e máximo, sem considerar os outliers)
    limite_inferior = np.min(data[data >= quartil_inferior - 1.5 * (quartil_superior - quartil_inferior)])  # Limite inferior
    limite_superior = np.max(data[data <= quartil_superior + 1.5 * (quartil_superior - quartil_inferior)])  # Limite superior
    
    # Cria o DataFrame com os valores calculados
    df = pd.DataFrame({
        'Quartil Inferior (Q1)': [quartil_inferior],
        'Mediana (Q2)': [mediana],
        'Quartil Superior (Q3)': [quartil_superior],
        'Limite Inferior': [limite_inferior],
        'Limite Superior': [limite_superior]
    })
    
    return df

# Calcula as estatísticas para cada conjunto de dados
df_convective_imerg_final = boxplot_stats(pps_convective_imerg_final)
df_nonconvective_imerg_final = boxplot_stats(pps_nonconvective_imerg_final)
df_all_imerg_final = boxplot_stats(pps_all_imerg_final)

df_convective_imerg_late = boxplot_stats(pps_convective_imerg_late)
df_nonconvective_imerg_late = boxplot_stats(pps_nonconvective_imerg_late)
df_all_imerg_late = boxplot_stats(pps_all_imerg_late)

df_convective_imerg_early = boxplot_stats(pps_convective_imerg_early)
df_nonconvective_imerg_early = boxplot_stats(pps_nonconvective_imerg_early)
df_all_imerg_early = boxplot_stats(pps_all_imerg_early)

df_convective_mvk = boxplot_stats(pps_convective_gsmap_mvk)
df_nonconvective_mvk = boxplot_stats(pps_nonconvective_gsmap_mvk)
df_all_mvk = boxplot_stats(pps_all_gsmap_mvk)

df_convective_nrt = boxplot_stats(pps_convective_gsmap_nrt)
df_nonconvective_nrt = boxplot_stats(pps_nonconvective_gsmap_nrt)
df_all_nrt = boxplot_stats(pps_all_gsmap_nrt)

df_convective_now = boxplot_stats(pps_convective_gsmap_now)
df_nonconvective_now = boxplot_stats(pps_nonconvective_gsmap_now)
df_all_now = boxplot_stats(pps_all_gsmap_now)

df_convective_gauge_mvk = boxplot_stats(pps_convective_gauge_mvk)
df_nonconvective_gauge_mvk = boxplot_stats(pps_nonconvective_gauge_mvk)
df_all_gauge_mvk = boxplot_stats(pps_all_gauge_mvk)

df_convective_gauge_nrt = boxplot_stats(pps_convective_gauge_nrt)
df_nonconvective_gauge_nrt = boxplot_stats(pps_nonconvective_gauge_nrt)
df_all_gauge_nrt = boxplot_stats(pps_all_gauge_nrt)

df_convective_gauge_now = boxplot_stats(pps_convective_gauge_now)
df_nonconvective_gauge_now = boxplot_stats(pps_nonconvective_gauge_now)
df_all_gauge_now = boxplot_stats(pps_all_gauge_now)

# Concatena os DataFrames

df_stats = pd.concat([df_convective_imerg_final, df_nonconvective_imerg_final, df_all_imerg_final,
                      df_convective_imerg_late, df_nonconvective_imerg_late, df_all_imerg_late,
                      df_convective_imerg_early, df_nonconvective_imerg_early, df_all_imerg_early,
                        df_convective_mvk, df_nonconvective_mvk, df_all_mvk,
                        df_convective_nrt, df_nonconvective_nrt, df_all_nrt,
                        df_convective_now, df_nonconvective_now, df_all_now,
                        df_convective_gauge_mvk, df_nonconvective_gauge_mvk, df_all_gauge_mvk,
                        df_convective_gauge_nrt, df_nonconvective_gauge_nrt, df_all_gauge_nrt,
                        df_convective_gauge_now, df_nonconvective_gauge_now, df_all_gauge_now],
                        keys=['IMERG-F', 'IMERG-L', 'IMERG-E', 'GSMaP-MVK', 'GSMaP-NRT', 'GSMaP-Now', 'GSMaP-MVK-G', 'GSMaP-NRT-G', 'GSMaP-Now-G'],
                        names=['Produto', 'Estatística'])

df_stats.to_csv('/home/milton/Documents/SBSR/boxplot_stats_pps.csv')'''


def percentile_metrics(pps, radar, product):
    percentiles = [0, 33, 66, 100]
    general_metrics_df = pd.DataFrame({'metrics': ['MAE', 'RMSE', 'Bias', 'Spearman Correlation', 
                                                   'Spearman Correlation p value']})
    for i in range(3):
        mask = (pps > np.percentile(pps, percentiles[i])) & (pps <= np.percentile(pps, percentiles[i+1]))
        metrics = aplicator.calculate_array_metrics(pps[mask], radar[mask])
        general_metrics_df[f'percentile_{percentiles[i]}_{percentiles[i+1]}'] = metrics
    percentile_values = ['percentil_values', 0, np.percentile(pps, 33), np.percentile(pps, 66)]
    new_row = pd.DataFrame([percentile_values], columns=general_metrics_df.columns)
    general_metrics_df = pd.concat([general_metrics_df, new_row], ignore_index=True)
    general_metrics_df.to_csv(f'/home/milton/Documents/SBSR/percentiles/{product}_percentile.csv', index=False)


percentile_metrics(pps_all_imerg_final,radar_all_imerg_final,'imerg_final_all')
percentile_metrics(pps_convective_imerg_final,radar_convective_imerg_final,'imerg_final_convective')
percentile_metrics(pps_nonconvective_imerg_final,radar_nonconvective_imerg_final,'imerg_final_nonconvective')

percentile_metrics(pps_all_imerg_late,radar_all_imerg_late,'imerg_late_all')
percentile_metrics(pps_convective_imerg_late,radar_convective_imerg_late,'imerg_late_convective')
percentile_metrics(pps_nonconvective_imerg_late,radar_nonconvective_imerg_late,'imerg_late_nonconvective')

percentile_metrics(pps_all_imerg_early,radar_all_imerg_early,'imerg_early_all')
percentile_metrics(pps_convective_imerg_early,radar_convective_imerg_early,'imerg_early_convective')
percentile_metrics(pps_nonconvective_imerg_early,radar_nonconvective_imerg_early,'imerg_early_nonconvective')

percentile_metrics(pps_all_gsmap_mvk,radar_all_gsmap_mvk,'mvk_all')
percentile_metrics(pps_convective_gsmap_mvk,radar_convective_gsmap_mvk,'mvk_convective')
percentile_metrics(pps_nonconvective_gsmap_mvk,radar_nonconvective_gsmap_mvk,'mvk_nonconvective')

percentile_metrics(pps_all_gsmap_nrt,radar_all_gsmap_nrt,'nrt_all')
percentile_metrics(pps_convective_gsmap_nrt,radar_convective_gsmap_nrt,'nrt_convective')
percentile_metrics(pps_nonconvective_gsmap_nrt,radar_nonconvective_gsmap_nrt,'nrt_nonconvective')

percentile_metrics(pps_all_gsmap_now,radar_all_gsmap_now,'now_all')
percentile_metrics(pps_convective_gsmap_now,radar_convective_gsmap_now,'now_convective')
percentile_metrics(pps_nonconvective_gsmap_now,radar_nonconvective_gsmap_now,'now_nonconvective')

percentile_metrics(pps_all_gauge_mvk,radar_all_gauge_mvk,'gauge_mvk_all')
percentile_metrics(pps_convective_gauge_mvk,radar_convective_gauge_mvk,'gauge_mvk_convective')
percentile_metrics(pps_nonconvective_gauge_mvk,radar_nonconvective_gauge_mvk,'gauge_mvk_nonconvective')

percentile_metrics(pps_all_gauge_nrt,radar_all_gauge_nrt,'gauge_nrt_all')
percentile_metrics(pps_convective_gauge_nrt,radar_convective_gauge_nrt,'gauge_nrt_convective')
percentile_metrics(pps_nonconvective_gauge_nrt,radar_nonconvective_gauge_nrt,'gauge_nrt_nonconvective')

percentile_metrics(pps_all_gauge_now,radar_all_gauge_now,'gauge_now_all')
percentile_metrics(pps_convective_gauge_now,radar_convective_gauge_now,'gauge_now_convective')
percentile_metrics(pps_nonconvective_gauge_now,radar_nonconvective_gauge_now,'gauge_now_nonconvective')


