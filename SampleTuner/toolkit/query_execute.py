import psycopg2
import time
import pandas as pd
import toolkit.tools
import toolkit.query_rewriter
import toolkit.query_parser


db_name = toolkit.tools.get_db_name()
query_data_dict = toolkit.tools.get_query_latency_result()
query_sample_dict = toolkit.tools.get_query_sample()
sample_statistic_dict = toolkit.tools.get_sample_statistic()
sample_ratio_dict = {}
query_selectivity_dict = {}
underlying_size_dict = {}


def exact_query_latency_result(query, sample=None):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    global query_data_dict, query_sample_dict, sample_statistic_dict
    query_e = query[:query.find(';')] if query.find(';') > 0 else query
    if sample:
        temp_tuple = tuple([query, sample.sample_name])
        if temp_tuple in query_sample_dict.keys():
            return query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
        else:
            sample_name = sample.sample_name
            temp_list = query_e.split(' ')
            temp_list[temp_list.index('FROM') + 1] = sample_name
            query_t = 'Explain Analyze ' + ' '.join(temp_list)
            query_r = ' '.join(temp_list)
            cur.execute(query_r)
            result = cur.fetchall()[0][0]
            result = float(result) if result else 0
    else:
        query_t = 'Explain Analyze ' + query_e
        query_r = query_e
        cur.execute(query_r)
        result = cur.fetchall()[0][0]
        result = float(result) if result else 0

    cur.execute(query_t)
    latency = float(cur.fetchall()[-1][0].split()[-2])

    if sample:
        query_sample_dict[tuple([query, sample.sample_name])] = tuple([latency, result])
    else:
        query_data_dict[query] = tuple([latency, result])

    conn.commit()
    conn.close()
    return latency, result


def exact_group_query_latency_result(query, sample=None):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    global query_data_dict, query_sample_dict, sample_statistic_dict
    query_e = query[:query.find(';')] if query.find(';') > 0 else query
    result = {}
    if sample:
        temp_tuple = tuple([query, sample.sample_name])
        if temp_tuple in query_sample_dict.keys():
            return query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
        else:
            sample_name = sample.sample_name
            table_name, _, _, _ = toolkit.query_parser.parser(query)
            query_e = query_e.replace(table_name, sample_name)
            query_t = 'Explain Analyze ' + query_e
            query_r = query_e
            cur.execute(query_r)
            result_list = cur.fetchall()
            for r in result_list:
                result[r[:-1]] = float(r[-1]) if r[-1] else 0
    else:
        query_t = 'Explain Analyze ' + query_e
        query_r = query_e
        cur.execute(query_r)
        result_list = cur.fetchall()
        for r in result_list:
            result[r[:-1]] = float(r[-1]) if r[-1] else 0

    cur.execute(query_t)
    latency = float(cur.fetchall()[-1][0].split()[-2])

    if sample:
        query_sample_dict[tuple([query, sample.sample_name])] = tuple([latency, result])
    else:
        query_data_dict[query] = tuple([latency, result])

    conn.commit()
    return latency, result


def query_statistic(query, sample):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    global sample_statistic_dict
    count, stddev = None, None
    query_e = query[:query.find(';')] if query.find(';') > 0 else query

    temp_list = query_e.split(' ')
    temp_list[temp_list.index('FROM') + 1] = sample.sample_name

    query_l = 'Explain Analyze ' + ' '.join(temp_list)
    cur.execute(query_l)
    latency = float(cur.fetchall()[-1][0].split()[-2])

    attr = temp_list[1]
    if attr != 'COUNT(*)':
        attr = attr.split('(')[-1].replace(')', '')
        temp_list.insert(temp_list.index('FROM'), ', STDDEV(' + attr + ')')
    else:
        attr = None
    query_r = ' '.join(temp_list)
    cur.execute(query_r)

    query_c = 'SELECT COUNT(*) FROM ' + sample.sample_name
    cur.execute(query_c)
    count = float(cur.fetchall()[0][0]) if count else 0

    if attr:
        query_result = cur.fetchall()
        result, stddev = query_result[0][0], query_result[0][1]
        stddev = float(stddev) if stddev else 0
        sample_statistic_dict[tuple([query, sample.sample_name])] = tuple([count, stddev])
    else:
        result = cur.fetchall()[0][0]
    result = float(result) if result else 0

    conn.commit()
    conn.close()
    return latency, result, count, stddev


def query_execute_data(workload):
    global query_data_dict
    u_l, u_r, w = [], [], []
    start = time.time()
    print('Executing query on underlying dataset ...')
    for query in workload:
        if query.find('GROUP BY') > 0 and query not in query_data_dict.keys():
            latency, result = exact_group_query_latency_result(query)
            w.append(query)
            u_l.append(latency)
            u_r.append(str(result))
        elif query not in query_data_dict.keys():
            latency, result = exact_query_latency_result(query)
            w.append(query)
            u_l.append(latency)
            u_r.append(result)
    print('The time cost of executing queries on underlying dataset (s): ', time.time() - start)
    if u_l:
        dataframe = pd.DataFrame({'query': w, 'latency': u_l, 'result': u_r})
        dataframe.to_csv(toolkit.tools.get_query_latency_result_name(), sep=',', mode='a')
    return query_data_dict


def query_selectivity(query):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    global query_selectivity_dict
    if query in query_selectivity_dict:
        selectivity = query_selectivity_dict[query]
    else:
        query = query[:query.find(';')] if query.find(';') > 0 else query
        temp_list = query.split(' ')
        temp_list[temp_list.index('FROM') - 1] = 'COUNT(*)'
        temp_list[temp_list.index('FROM') + 1] += '_selectivity'

        data_size = underlying_size(temp_list[temp_list.index('FROM') + 1])
        sql = ' '.join(temp_list)
        cur.execute(sql)
        subset_size = int(cur.fetchall()[0][0])
        selectivity = subset_size / data_size
        query_selectivity_dict[query] = selectivity
        conn.commit()
    conn.close()
    return selectivity


def underlying_size(table_name):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    global underlying_size_dict
    if table_name in underlying_size_dict:
        size = underlying_size_dict[table_name]
    else:
        sql = 'SELECT number FROM tablemeta WHERE tablename = ' + "'" + table_name + "'"
        cur.execute(sql)
        size = int(cur.fetchall()[0][0])
        underlying_size_dict[table_name] = size
        conn.commit()
    conn.close()
    return size


def sample_ratio(sample_name):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    global sample_ratio_dict
    if sample_name in sample_ratio_dict:
        ratio = sample_ratio_dict[sample_name]
    else:
        sql = 'SELECT ratio FROM sampleratio WHERE samplename = ' + "'" + sample_name + "'"
        cur.execute(sql)
        ratio = float(cur.fetchall()[0][0])
        sample_ratio_dict[sample_name] = ratio
        conn.commit()
    conn.close()
    return ratio


def query_sample_statistics(workload, abstract_sample_list):
    global query_data_dict, sample_statistic_dict
    w_1, w_2, s_1, s_2, s_l, s_r, s_c, s_s = [], [], [], [], [], [], [], []
    print('Execute queries on samples...')
    start = time.time()
    for query in workload:
        samples = toolkit.query_rewriter.choose_candidate_active_sample(query, abstract_sample_list)
        for sample in samples:
            temp_tuple = tuple([query, sample.sample_name])
            if temp_tuple not in query_sample_dict.keys():
                latency, result = exact_query_latency_result(query, sample)
                w_1.append(query)
                s_1.append(sample.sample_name)
                s_l.append(latency)
                s_r.append(result)
            if temp_tuple not in sample_statistic_dict.keys():
                _, _, count, stddev = query_statistic(query, sample)
                w_2.append(query)
                s_2.append(sample.sample_name)
                s_c.append(count)
                s_s.append(stddev)
    if s_l:
        dataframe = pd.DataFrame({'query': w_1, 'sample': s_1, 'latency': s_l, 'result': s_r})
        dataframe.to_csv(toolkit.tools.get_query_sample_name(), sep=',', mode='a')
    if s_c:
        dataframe = pd.DataFrame({'query': w_2, 'sample': s_2, 'count': s_c, 'stddev': s_s})
        dataframe.to_csv(toolkit.tools.get_sample_statistic_name(), sep=',', mode='a')
    print('The time cost of execute queries on sample: ', time.time() - start)
