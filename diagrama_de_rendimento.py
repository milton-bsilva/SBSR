import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def csi_from_far_pod(far,pod): #CSI = 1/(1/(1-FAR) + (1/POD) - 1)
    CSI = 1/(1/(1-far) + (1/pod) - 1)
    return CSI

##########
table_imerg=pd.read_parquet('/media/milton/Data/Data/Tables/qualitatives_imerg_000.parquet')
table_imerg=table_imerg.loc[table_imerg['timestamp']>'2019-07-01']

far_final=table_imerg['False Positive final'].sum()/(table_imerg['True Positive final'].sum()+table_imerg['False Positive final'].sum())
sr_final=1-far_final
pod_final=table_imerg['True Positive final'].sum()/(table_imerg['True Positive final'].sum()+table_imerg['False Negative final'].sum())
bias_final=(pod_final/sr_final)

far_late=table_imerg['False Positive late'].sum()/(table_imerg['True Positive late'].sum()+table_imerg['False Positive late'].sum())
sr_late=1-far_late
pod_late=table_imerg['True Positive late'].sum()/(table_imerg['True Positive late'].sum()+table_imerg['False Negative late'].sum())
bias_late=(pod_late/sr_late)

far_early=table_imerg['False Positive early'].sum()/(table_imerg['True Positive early'].sum()+table_imerg['False Positive early'].sum())
sr_early=1-far_early
pod_early=table_imerg['True Positive early'].sum()/(table_imerg['True Positive early'].sum()+table_imerg['False Negative early'].sum())
bias_early=(pod_early/sr_early)

############
table_gsmap=pd.read_parquet('/media/milton/Data/Data/Tables/qualitatives_gsmap_abc.parquet')
table_gsmap=table_gsmap.loc[table_gsmap['timestamp']>'2019-07-01']
far_mvk=table_gsmap['False Positive MVK'].sum()/(table_gsmap['True Positive MVK'].sum()+table_gsmap['False Positive MVK'].sum())
sr_mvk=1-far_mvk
pod_mvk=table_gsmap['True Positive MVK'].sum()/(table_gsmap['True Positive MVK'].sum()+table_gsmap['False Negative MVK'].sum())
bias_mvk=(pod_mvk/sr_mvk)

far_nrt=table_gsmap['False Positive NRT'].sum()/(table_gsmap['True Positive NRT'].sum()+table_gsmap['False Positive NRT'].sum())
sr_nrt=1-far_nrt
pod_nrt=table_gsmap['True Positive NRT'].sum()/(table_gsmap['True Positive NRT'].sum()+table_gsmap['False Negative NRT'].sum())
bias_nrt=(pod_nrt/sr_nrt)

far_gauge_mvk=table_gsmap['False Positive GAUGE MVK'].sum()/(table_gsmap['True Positive GAUGE MVK'].sum()+table_gsmap['False Positive GAUGE MVK'].sum())
sr_gauge_mvk=1-far_gauge_mvk
pod_gauge_mvk=table_gsmap['True Positive GAUGE MVK'].sum()/(table_gsmap['True Positive GAUGE MVK'].sum()+table_gsmap['False Negative GAUGE MVK'].sum())
bias_gauge_mvk=(pod_gauge_mvk/sr_gauge_mvk)

far_gauge_nrt=table_gsmap['False Positive GAUGE NRT'].sum()/(table_gsmap['True Positive GAUGE NRT'].sum()+table_gsmap['False Positive GAUGE NRT'].sum())
sr_gauge_nrt=1-far_gauge_nrt
pod_gauge_nrt=table_gsmap['True Positive GAUGE NRT'].sum()/(table_gsmap['True Positive GAUGE NRT'].sum()+table_gsmap['False Negative GAUGE NRT'].sum())
bias_gauge_nrt=(pod_gauge_nrt/sr_gauge_nrt)

########
table_gsmapnow=pd.read_parquet('/media/milton/Data/Data/Tables/qualitatives_gsmapnow_000.parquet')
table_gsmapnow=table_gsmapnow.loc[table_gsmapnow['timestamp']>'2019-07-01']
far_now=table_gsmapnow['False Positive NOW'].sum()/(table_gsmapnow['True Positive NOW'].sum()+table_gsmapnow['False Positive NOW'].sum())
sr_now=1-far_now
pod_now=table_gsmapnow['True Positive NOW'].sum()/(table_gsmapnow['True Positive NOW'].sum()+table_gsmapnow['False Negative NOW'].sum())
bias_now=(pod_now/sr_now)

table_gsmapnow_gauge=pd.read_parquet('/media/milton/Data/Data/Tables/qualitatives_gsmapnow_gauge.parquet')
table_gsmapnow_gauge=table_gsmapnow_gauge.loc[table_gsmapnow_gauge['timestamp']>'2019-07-01']
far_now_gauge=table_gsmapnow_gauge['False Positive NOW'].sum()/(table_gsmapnow_gauge['True Positive NOW'].sum()+table_gsmapnow_gauge['False Positive NOW'].sum())
sr_now_gauge=1-far_now_gauge
pod_now_gauge=table_gsmapnow_gauge['True Positive NOW'].sum()/(table_gsmapnow_gauge['True Positive NOW'].sum()+table_gsmapnow_gauge['False Negative NOW'].sum())
bias_now_gauge=(pod_now_gauge/sr_now_gauge)


#print bias
print('Bias IMERG Final:',bias_final)
print('Bias IMERG Late:',bias_late)
print('Bias IMERG Early:',bias_early)
print('Bias GSMaP MVK:',bias_mvk)
print('Bias GSMaP NRT:',bias_nrt)
print('Bias GSMaP Now:',bias_now)
print('Bias GSMaP Gauge MVK:',bias_gauge_mvk)
print('Bias GSMaP Gauge NRT:',bias_gauge_nrt)

################
far_range,pod_range=np.meshgrid(np.arange(0.0001,1.005,.005),np.arange(0.0001,1.005,.005))
csi_range=csi_from_far_pod(far_range,pod_range)

thetas = [.5,1,1.5,2,4] # Cinco valores de ângulo entre 0 e 90 graus


fig,ax=plt.subplots(figsize=(10,10))
cf=ax.contour(1-far_range,pod_range,csi_range,levels=np.arange(0,1.1,.1),cmap='hsv',zorder=2)
cf.clabel(fmt='%1.1f')
ax.scatter(sr_final,pod_final,c='orange',marker='s',edgecolor='black',label='IMERG-F',s=200,zorder=100)
ax.scatter(sr_late,pod_late,c='orange',marker='^',edgecolor='black',label='IMERG-L',s=200,zorder=100)
ax.scatter(sr_early,pod_early,c='orange',marker='o', edgecolor='black',label='IMERG-E',s=200,zorder=100)
ax.scatter(sr_mvk,pod_mvk,c='green',marker='s', edgecolor='black',label='GSMaP-MVK',s=200,zorder=100)
ax.scatter(sr_nrt,pod_nrt,c='green',marker='^', edgecolor='black',label='GSMaP-NRT',s=200,zorder=100)
ax.scatter(sr_now,pod_now,c='green',marker='o', edgecolor='black',label='GSMaP-Now',s=200,zorder=100)
ax.scatter(sr_gauge_mvk,pod_gauge_mvk,c='pink',marker='s',edgecolor='black',label='GSMaP-MVK-G',s=200,zorder=100)
ax.scatter(sr_gauge_nrt,pod_gauge_nrt,c='pink',marker='^',edgecolor='black',label='GSMaP-NRT-G',s=200,zorder=100)
ax.scatter(sr_now_gauge,pod_now_gauge,c='pink',marker='o',edgecolor='black',label='GSMaP-Now-G',s=200,zorder=100)

far_range_first = (1-far_range)[0]  # Extract the first element
ay2=ax.twinx()
ay2.set_yticks([0.4975],labels=['0.5'])
ax2=ax.twiny()
ax2.set_xticks([1,.67,.5,.25],labels=[1,1.5,2,4])
for m in thetas:    
    y=m*(1-far_range)[0]
    ax.plot((1-far_range)[0],y,'--',c='k',zorder=1) #plot bias


ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_ylabel('POD')
ax.set_xlabel('TS (1-FAR)')
legend = ax.legend(frameon=True,loc=2)  # zorder para a legenda
legend.get_frame().set_alpha(1)  # Caixa da legenda sem transparência

plt.savefig('/home/milton/Documents/SBSR/Figuras/csi.png',dpi=300,bbox_inches='tight')