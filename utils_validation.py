import numpy as np
from scipy import stats
import swifter
import pandas as pd

class MetricsCalculator:
    @staticmethod
    def _remove_nan(pps, radar):
        valid_indices = ~np.isnan(radar)
        return pps[valid_indices], radar[valid_indices]

    def calculate_MAE(self, pps, radar): #1
        return np.mean(np.abs(pps - radar))

    def calculate_NMAE(self, pps, radar): #1
        return np.mean(np.abs(pps - radar))/np.mean(radar)

    def calculate_RMSE(self, pps, radar): #2
        return np.sqrt(np.mean((pps - radar) ** 2))

    def calculate_MAPE(self, pps,radar): #3
        pps=pps[radar!=0]
        radar=radar[radar!=0]
        #pps, radar = self._remove_nan(pps, radar)
        MAPE = np.mean(np.abs((radar - pps) / radar)) * 100
        return MAPE

    def calculate_bias(self, pps, radar): #4
        bias = np.sum(pps - radar) / len(radar)
        return bias

    def calculate_far(self, pps, radar,limiar=0): #5
        true_p = np.logical_and((radar > limiar), (pps > limiar))
        false_p = np.logical_and((radar <= limiar), (pps > limiar))
        far = np.sum(false_p) / (np.sum(true_p) + np.sum(false_p))
        return far


    def calculate_pod(self, pps, radar,limiar=0): #5
        true_p = np.logical_and((radar > limiar), (pps > limiar))
        false_n = np.logical_and((radar > limiar), (pps <= limiar))
        pod = np.sum(true_p) / (np.sum(true_p) + np.sum(false_n))
        return pod
    
    def calculate_csi(self, pps, radar,limiar=0): #5
        true_p = np.logical_and((radar > limiar), (pps > limiar))
        false_p = np.logical_and((radar <= limiar), (pps > limiar))
        false_n = np.logical_and((radar > limiar), (pps <= limiar))
        csi = np.sum(true_p) / (np.sum(true_p) + np.sum(false_p) + np.sum(false_n))
        return csi

    def calculate_nse(self, pps, radar): #6
        return 1 - np.nansum((radar - pps) * 2) / np.nansum((radar - np.nanmean(pps)) * 2,axis=0)


    def calculate_spearman_correlation(self, pps, radar): #7
        return stats.spearmanr(pps, radar)

    def calculate_kendall_tau(self, pps, radar): #8
        tau, p_value = stats.kendalltau(radar, pps)
        # p-value < 0.05 is significant
        return tau, p_value

    def calculate_POI(self, pps, radar):
        # pdf normalized
        radar = radar[~np.isnan(radar)]
        pdf1, bins1 = np.histogram(radar, bins=10, density=True)
        pdf2, bins2 = np.histogram(pps, bins=10, density=True)
        overlap = np.sum(np.minimum(pdf1 * np.diff(bins1), pdf2 * np.diff(bins2)))
        return overlap

    def calculate_Kolmogorov_Smirnov(self, pps, radar):
        radar = radar[~np.isnan(radar)]
        # p-value > 0.05 means that the two distributions are similar
        return stats.ks_2samp(radar, pps)

    def calculate_rmsd(self, pps, radar):
        differences = radar - pps
        squared_differences = np.square(differences)
        mean_squared_differences = np.nanmean(squared_differences, axis=0)
        rmsd = np.sqrt(mean_squared_differences)
        return rmsd
    
    def calculate_quantiles(self, pps, radar):
        quantiles=np.arange(5,100,5)
        qqline=np.zeros((2,quantiles.size))
        for i,j in enumerate(quantiles):
            qqline[0,i]=np.percentile(pps,j)
            qqline[0,i]=np.percentile(radar,j)
        return qqline

class MetricsApplicator:
    def __init__(self, calculator):
        self.calculator = calculator
        self.columns = ['MAE', 'RMSE', 'Bias','Spearman Correlation', 'Spearman Correlation p value']

    def calculate_array_metrics(self, array_pps, array_radar):
        array_metrics = np.zeros(5)
        array_metrics[0] = self.calculator.calculate_MAE(array_pps, array_radar)
        array_metrics[1] = self.calculator.calculate_RMSE(array_pps, array_radar)
        array_metrics[2] = self.calculator.calculate_bias(array_pps, array_radar)
        result_spearman = self.calculator.calculate_spearman_correlation(array_pps, array_radar)
        array_metrics[3] = result_spearman[0]
        array_metrics[4] = result_spearman[1]
        return array_metrics


    def compute_metrics(self, df_):
        n = len(df_)
        array_metrics = np.zeros((n, 5))
        for i in range(n):
            pps_values = df_['array_values'].iloc[i]
            radar_values = df_['radar_values'].iloc[i]
            array_metrics[i] = self.calculate_array_metrics(pps_values, radar_values)
        df_metrics = pd.DataFrame(array_metrics, columns=self.columns)
        df_metrics['classification'] = df_['classification'].values
        return df_metrics


    def calculate_hourly_metrics(self, array_pps, array_radar,array_hour):
        
        hourly_statistics = np.zeros((len(np.unique(array_hour)), 5))

        for i,j in enumerate(np.unique(array_hour)):
            mask = array_hour == j
            array_pps_hour = array_pps[mask]
            array_radar_hour = array_radar[mask]
            hourly_statistics[i] = self.calculate_array_metrics(array_pps_hour, array_radar_hour)
        return pd.DataFrame(hourly_statistics, columns=self.columns)

    def calculate_levels_metric(self, array_pps, array_radar):
        levels = [np.percentile(array_radar, i) for i in [0, 33, 66, 100]]
        levels_statistics = np.zeros((3, 5))
        for i in range(3):
            mask = (array_radar > levels[i]) & (array_radar <= levels[i + 1])
            array_pps_level = array_pps[mask]
            array_radar_level = array_radar[mask]
            levels_statistics[i] = self.calculate_array_metrics(array_pps_level, array_radar_level)
        return pd.DataFrame(levels_statistics, columns=self.columns)

    def concatenate_values_types(self, df_):
        df_['hour'] = df_.apply(lambda x: [x['hour']] * len(x['array_values']), axis=1)
        convective_df_ = df_[df_['classification'] == 1]
        nonconvective_df_ = df_[df_['classification'] == 0]
        mixed_df_ = df_[df_['classification'] == 2]
        pps_convective = np.concatenate(convective_df_['array_values'].values).astype(float)
        radar_convective = np.concatenate(convective_df_['radar_values'].values).astype(float)
        hour_convective = np.concatenate(convective_df_['hour'].values).astype(float)
        pps_nonconvective = np.concatenate(nonconvective_df_['array_values'].values).astype(float)
        radar_nonconvective = np.concatenate(nonconvective_df_['radar_values'].values).astype(float)
        hour_nonconvective = np.concatenate(nonconvective_df_['hour'].values).astype(float)
        pps_mixed = np.concatenate(mixed_df_['array_values'].values).astype(float)
        radar_mixed = np.concatenate(mixed_df_['radar_values'].values).astype(float)
        hour_mixed = np.concatenate(mixed_df_['hour'].values).astype(float)
        return pps_convective, radar_convective, hour_convective,pps_nonconvective, radar_nonconvective, hour_nonconvective,pps_mixed, radar_mixed, hour_mixed

    def concatenate_convec_and_nonconvec(self, df_):
        array_values = np.concatenate(df_['array_values'].values).astype(float)
        radar_values = np.concatenate(df_['radar_values'].values).astype(float)
        classification = np.concatenate(df_['classification'].values)
        mask = ~np.isnan(radar_values)
        array_values = array_values[mask]
        radar_values = radar_values[mask]
        classification = classification[mask]
        mask_convective = (classification == 1)
        array_values_convective = array_values[mask_convective]
        radar_values_convective = radar_values[mask_convective]
        array_values_nonconvective = array_values[~mask_convective]
        radar_values_nonconvective = radar_values[~mask_convective]
        return (array_values_convective, radar_values_convective, 
                array_values_nonconvective, radar_values_nonconvective,
                array_values, radar_values)

    def concatenate_array_positions(self, df_):
        array_x = np.concatenate(df_['array_x'].values).astype(float)
        array_y = np.concatenate(df_['array_y'].values).astype(float)
        classification = np.concatenate(df_['classification'].values)[:len(array_x)]
        radar_values = np.concatenate(df_['radar_values'].values).astype(float)
        mask = ~np.isnan(radar_values)
        array_x = array_x[mask]
        array_y = array_y[mask]
        classification = classification[mask]
        radar_values = radar_values[mask]
        mask_convective = (classification == 1)
        array_x_convective = array_x[mask_convective]
        array_y_convective = array_y[mask_convective]
        #non convective
        array_x_nonconvective = array_x[~mask_convective]
        array_y_nonconvective = array_y[~mask_convective]
        return (array_x_convective, array_y_convective,array_x_nonconvective, array_y_nonconvective,array_x, array_y)


# Exemplo de uso
if __name__ == "__main__":
    obsSTD = 1
    s = [1.275412605, 1.391302157, 1.424314937]
    s1 = [0.980035327, 0.997244197, 1.003002031]
    r = [0.572272, 0.533529, 0.477572]
    r1 = [0.82, 0.72, 0.8]
    l = ['A', 'B', 'C']
    l1 = ['A', 'B', 'C']
    fname = 'TaylorDiagram.jpg'

    dia = TaylorDiagram(obsSTD)
    dia.srl(s, s1, r, r1, l, l1, fname)
    pps = np.array([0.2, 0.5, 0.6, np.nan, 0.7, 0.8])
    radar = np.array([0.3, 0.6, 0.5, 0.4, np.nan, 0.9])

    calculator = MetricsCalculator()
    print("FAR:", calculator.calculate_far(pps, radar))
    print("Bias:", calculator.calculate_bias(pps, radar))
    print("Pearson Correlation:", calculator.calculate_pearson_correlation(pps, radar))
    print("Kendall Tau:", calculator.calculate_kendall_tau(pps, radar))
    print("RMSE:", calculator.calculate_RMSE(pps, radar))
    print("POI:", calculator.calculate_POI(pps, radar))
    print("Kolmogorov-Smirnov:", calculator.calculate_Kolmogorov_Smirnov(pps, radar))