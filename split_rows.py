import pandas as pd
import duckdb
import numpy as np
import utils_validation

def save_parquet_by_genesis(row,savedir):
    radar_values = row['radar_values']
    mask = ~np.isnan(radar_values)
    array_x = row['array_x'][mask]
    array_y = row['array_y'][mask]
    radar_values = row['radar_values'][mask]
    array_values = row['array_values'][mask]
    classification = row['classification'][mask]
    timestamp_str = str(row['timestamp']).replace(' ', 'T')
    savename = f"{savedir}{str(int(row['uid'])).zfill(6)}_{timestamp_str}.parquet"
    df = pd.DataFrame({
        'array_x': array_x,
        'array_y': array_y,
        'radar_values': radar_values,
        'array_values': array_values,
        'classification': classification,})
    df.to_parquet(savename)
    array_properties = np.array([row['size'], row['duration'], row['lifetime']]).astype(int)
    array_properties.to_file(savename.replace('parquet', 'dat'))



product='track_imerg_final'

query = f"""
WITH filtered_base AS (
    SELECT *
    FROM
        parquet_scan('/media/milton/Data/Data/Tracks/{product}/trackingtable/*.parquet', union_by_name=True)
    WHERE
        uid IN (
            SELECT
                uid
            FROM
                parquet_scan('/media/milton/Data/Data/Tracks/{product}/trackingtable/*.parquet', union_by_name=True)
            GROUP BY
                uid
            HAVING
                COUNT(DISTINCT CASE WHEN status NOT IN ('NEW', 'CON') THEN status END) = 0
                AND COUNT(*) > 3))
SELECT *
FROM
    filtered_base
"""

with duckdb.connect(database=':memory:', read_only=False) as con:
    base = con.sql(query).df()

#filter timestamp >= 2019-07-01 00:00 and <= 2024-04-30  23:00

query = """
SELECT *
FROM
    base
WHERE
    timestamp >= '2019-07-01 00:00:00' AND timestamp <= '2024-04-30 23:00:00'
"""

with duckdb.connect(database=':memory:', read_only=False) as con:
    base_filt = con.sql(query).df()


query_begin="""'
SELECT
    timestamp,
    uid,
    size,
    duration,
    lifetime,
    array_y,
    array_x,
    array_values,
    radar_values,
    classification,
FROM
    base_filt
WHERE
    genesis = 1
"""

with duckdb.connect(database=':memory:', read_only=False) as con:
    df_begin = con.sql(query_begin).df()

for idx, row in df_begin.iterrows():
    radar_values = row['radar_values']
    classification = row['classification']
    #if all radar_values are nan
    if isinstance(radar_values, np.ma.MaskedArray) or not isinstance(radar_values, np.ndarray) or isinstance(classification, float):
        continue
    save_parquet_by_genesis(row,f'/media/milton/Data/Data/Validation/{product}/save_begin/')

del df_begin


query_end="""
SELECT
    timestamp,
    uid,
    size,
    duration,
    lifetime,
    array_y,
    array_x,
    array_values,
    radar_values,
    classification,
FROM
    base_filt
WHERE
    genesis = -1
"""

with duckdb.connect(database=':memory:', read_only=False) as con:
    df_end = con.sql(query_end).df()

for idx, row in df_end.iterrows():
    radar_values = row['radar_values']
    classification = row['classification']
    #if all radar_values are nan
    if isinstance(radar_values, np.ma.MaskedArray) or not isinstance(radar_values, np.ndarray) or isinstance(classification, float):
        continue
    save_parquet_by_genesis(row,f'/media/milton/Data/Data/Validation/{product}/save_end/')

del df_end

query_develop="""
SELECT
    timestamp,
    uid,
    size,
    duration,
    lifetime,
    array_y,
    array_x,
    array_values,
    radar_values,
    classification,
FROM
    base_filt
WHERE
    genesis = 0
"""

with duckdb.connect(database=':memory:', read_only=False) as con:
    df_develop = con.sql(query_develop).df()

for idx, row in df_develop.iterrows():
    radar_values = row['radar_values']
    classification = row['classification']
    #if all radar_values are nan
    if isinstance(radar_values, np.ma.MaskedArray) or not isinstance(radar_values, np.ndarray) or isinstance(classification, float):
        continue
    save_parquet_by_genesis(row,f'/media/milton/Data/Data/Validation/{product}/save_develop/')

del df_develop


query="""
WITH lifetime_size_diff AS (
    SELECT
        *,
        (1.0 / size) * ((size - LAG(size, 1) OVER (PARTITION BY uid ORDER BY lifetime)) /
        (lifetime - LAG(lifetime, 1) OVER (PARTITION BY uid ORDER BY lifetime))) AS expansion
    FROM
        base
),
min_expansion AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY uid ORDER BY ABS(expansion) ASC) AS rn
    FROM
        lifetime_size_diff
    WHERE
        genesis = 0
)
SELECT
    timestamp,
    uid,
    size,
    duration,
    lifetime,
    array_y,
    array_x,
    array_values,
    radar_values,
    classification,
    expansion
FROM
    min_expansion
WHERE
    rn = 1;
"""

with duckdb.connect() as con:
    base_exp = con.execute(query).df()

#filter date
query = """
SELECT *
FROM
    base_exp
WHERE
    timestamp >= '2019-07-01 00:00:00' AND timestamp <= '2024-04-30 23:00:00'
"""

with duckdb.connect() as con:
    df_exp = con.execute(query).df()

for idx, row in df_exp.iterrows():
    radar_values = row['radar_values']
    classification = row['classification']
    #if all radar_values are nan
    if isinstance(radar_values, np.ma.MaskedArray) or not isinstance(radar_values, np.ndarray) or isinstance(classification, float):
        continue
    save_parquet_by_genesis(row,f'/media/milton/Data/Data/Validation/{product}/save_expansion/')


############# analise de dados

def calculate_boxplot_stats(data):
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    return lower_fence, q1, median, q3, upper_fence



calculator = utils_validation.MetricsCalculator()
aplicator = utils_validation.MetricsApplicator(calculator)

query_statistic = """
SELECT
array_values,
radar_values,
classification
FROM parquet_scan(f'{load_file}*.parquet', union_by_name=True);
"""

phases=['begin', 'develop', 'end']
for phase in phases:
    load_file=f'/media/milton/Data/Data/Validation/{product}/save_{phase}/'

    with duckdb.connect() as con:
        base_statistic = con.execute(query).df()

    pps_stratiform=base_statistic['array_values'][base_statistic['classification']==0].values
    pps_convective=base_statistic['array_values'][base_statistic['classification']==1].values

    radar_stratiform=base_statistic['radar_values'][base_statistic['classification']==0].values
    radar_convective=base_statistic['radar_values'][base_statistic['classification']==1].values

    pps_all=base_statistic['array_values'].values
    radar_all=base_statistic['radar_values'].values

    general_metrics_df = pd.DataFrame({'metrics': ['MAE', 'RMSE', 'Bias', 'Spearman Correlation',
        'Spearman Correlation p value'],
        'convective': aplicator.calculate_array_metrics(pps_convective, radar_convective),
        'stratiform': aplicator.calculate_array_metrics(pps_stratiform,radar_stratiform),
        'all': aplicator.calculate_array_metrics(pps_all, radar_all),
        'boxplot': ['lower_fence', 'q1', 'median', 'q3', 'upper_fence'],
        'convective_pps': aplicator.calculate_boxplot_stats(pps_convective),
        'stratiform_pps': aplicator.calculate_boxplot_stats(pps_stratiform),
        'all_pss': aplicator.calculate_boxplot_stats(pps_all),
        'convective_radar': aplicator.calculate_boxplot_stats(radar_convective),
        'stratiform_radar': aplicator.calculate_boxplot_stats(radar_stratiform),
        'all_radar': aplicator.calculate_boxplot_stats(radar_all)
        })
    general_metrics_df.to_csv(f'/media/milton/Data/Data/Validation/{product}/metrics_{phase}.csv')
