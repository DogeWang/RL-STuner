import sys

sys.path.append('../')

import psycopg2
import pandas as pd
import toolkit.tools


db_name = toolkit.tools.get_db_name()


def delete():
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    cur.execute("select * from pg_tables where schemaname='public'")
    attribute_list = cur.fetchall()
    materialized_samples = toolkit.tools.get_samples()
    query_sample_dict = toolkit.tools.get_query_sample()
    sample_statistic_dict = toolkit.tools.get_sample_statistic()
    judge = False
    for i in attribute_list:
        sample_name = i[1]
        if 'uniform' in sample_name or 'stratified' in sample_name:
            judge = True
            cur.execute('DROP TABLE IF EXISTS ' + sample_name)
            cur.execute('DELETE FROM sampleratio WHERE samplename = ' + "'" + sample_name + "'")
            if sample_name in list(materialized_samples.keys()):
                del materialized_samples[sample_name]
            for key in list(query_sample_dict.keys()):
                if sample_name in key:
                    del query_sample_dict[key]
            for key in list(sample_statistic_dict.keys()):
                if sample_name in key:
                    del sample_statistic_dict[key]

    file = open(toolkit.tools.get_samples_name(), 'w')
    for sample_name in materialized_samples:
        file.write(sample_name + ' ' + str(materialized_samples[sample_name]) + '\n')

    w_1, w_2, s_1, s_2, s_l, s_r, s_c, s_s = [], [], [], [], [], [], [], []

    for key, item in query_sample_dict.items():
        w_1.append(key[0])
        s_1.append(key[1])
        s_l.append(item[0])
        s_r.append(item[1])

    dataframe = pd.DataFrame({'query': w_1, 'sample': s_1, 'latency': s_l, 'result': s_r})
    dataframe.to_csv(toolkit.tools.get_query_sample_name(), sep=',', mode='w')

    for key, item in sample_statistic_dict.items():
        w_2.append(key[0])
        s_2.append(key[1])
        s_c.append(item[0])
        s_s.append(item[1])
    dataframe = pd.DataFrame({'query': w_2, 'sample': s_2, 'count': s_c, 'stddev': s_s})
    dataframe.to_csv(toolkit.tools.get_sample_statistic_name(), sep=',', mode='w')

    if judge:
        print('Delete samples.')
    else:
        print('No samples.')
    conn.commit()
    conn.close()


delete()