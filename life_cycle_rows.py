import pandas as pd
import duckdb
import numpy as np
import matplotlib.pyplot as plt
product='track_imerg_final'

query = """
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
SELECT *,
        (1.0 / size) * ((size - LAG(size, 1) OVER (PARTITION BY uid ORDER BY lifetime)) /
        (lifetime - LAG(lifetime, 1) OVER (PARTITION BY uid ORDER BY lifetime))) AS expansion
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
    basenew = con.sql(query).df()

def define_class(cl):
    type_cloud = np.sum(cl) / len(cl)
    if type_cloud == 1:
        return 1
    elif type_cloud == 0:
        return 0
    else:
        return 2

def sum_precipitation(rain):
    return np.nansum(rain)

basenew = basenew.dropna(subset=['classification'])
basenew['classification'] = basenew['classification'].apply(define_class)
basenew['rain']=basenew['array_values'].apply(sum_precipitation)


#separate by classi

query_stratiform = f"""
WITH expansion_df AS (
    SELECT
        *,
    FROM
        basenew
    WHERE
        classification = 0)                parquet_scan('/home/milton/Documentos/trackingtable/*.parquet', union_by_name=True)

SELECT duration,
       lifetime,
       AVG(expansion) AS avg_expansion,
       AVG(rain) AS avg_rain
FROM expansion_df
GROUP BY duration, lifetime
ORDER BY duration, lifetime;
"""

with duckdb.connect() as con:
    diff_strat = con.execute(query_stratiform).df()

query_convective = f"""
WITH expansion_df AS (
    SELECT
        *,
    FROM
        basenew
    WHERE
        classification = 1)
SELECT duration,
       lifetime,
       AVG(expansion) AS avg_expansion,
       AVG(rain) AS avg_rain
FROM expansion_df
GROUP BY duration, lifetime
ORDER BY duration, lifetime;
"""

with duckdb.connect() as con:
    diff_conv = con.execute(query_convective).df()

query_mixes = f"""
WITH expansion_df AS (
    SELECT
        *,
    FROM
        basenew
    WHERE
        classification = 2)
SELECT duration,
         lifetime,
         AVG(expansion) AS avg_expansion,
         AVG(rain) AS avg_rain  
FROM expansion_df
GROUP BY duration, lifetime
ORDER BY duration, lifetime;
"""

with duckdb.connect() as con:
    diff_mix = con.execute(query_mixes).df()

durations=np.unique(basenew['duration'])

colors=['blue','orange','green','red','purple','brown']
fiz, ax = plt.subplots(2, 3, figsize=(10, 10))
for i,dur in enumerate(durations):
  ax[0,0].plot(diff_strat[diff_strat['duration']==dur]['lifetime'],diff_strat[diff_strat['duration']==dur]['avg_expansion'],label=dur,color=colors[i])
  ax[0,1].plot(diff_conv[diff_conv['duration']==dur]['lifetime'],diff_conv[diff_conv['duration']==dur]['avg_expansion'],label=dur,color=colors[i])
  ax[0,2].plot(diff_mix[diff_mix['duration']==dur]['lifetime'],diff_mix[diff_mix['duration']==dur]['avg_expansion'],label=dur,color=colors[i])
  ax[1,0].plot(diff_strat[diff_strat['duration']==dur]['lifetime'],diff_strat[diff_strat['duration']==dur]['avg_rain'],label=dur,color=colors[i])
  ax[1,1].plot(diff_conv[diff_conv['duration']==dur]['lifetime'],diff_conv[diff_conv['duration']==dur]['avg_rain'],label=dur,color=colors[i])
  ax[1,2].plot(diff_mix[diff_mix['duration']==dur]['lifetime'],diff_mix[diff_mix['duration']==dur]['avg_rain'],label=dur,color=colors[i])

ax[0,0].set_title('Sistemas estratiformes')
ax[0,1].set_title('Sistemas convectivos')
ax[0,2].set_title('Sistemas mistos')

ax[0,0].set_ylabel('Expansão Média')
ax[1,0].set_ylabel('Taxa de precipitação média')

ax[0,0].legend()
ax[1,0].set_xlabel('Tempo de vida')
ax[1,1].set_xlabel('Tempo de vida')
ax[1,2].set_xlabel('Tempo de vida')

plt.savefig(f'/media/milton/Data/Data/Validation/{product}/expansion_rain.png',dpi=300,bbox_inches='tight')