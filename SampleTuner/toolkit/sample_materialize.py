import psycopg2
import hashlib
import itertools
import json
import time
import os
import toolkit.tools
import toolkit.query_execute
import toolkit.query_parser


db_name = toolkit.tools.get_db_name()
query_to_sample_dict = {}
sample_name_set = set()
sample_list = list()
cwd = os.getcwd()


class Sample:
    def __init__(self, sample_name, sample_size, target_table, sample_type, target_dataset,
                 filter_condition, group_list, sample_cost=None, materialization=False):
        self.sample_name = sample_name
        self.sample_size = sample_size
        self.target_table = target_table
        self.sample_type = sample_type
        self.target_dataset = target_dataset
        self.filter_condition = filter_condition
        self.group_list = group_list
        self.sample_cost = sample_cost
        self.materialization = materialization

    def equal(self, sample):
        if self.sample_name == sample.sample_name:
            return True
        return False


# construct sample
def construct_sample(table_name, sample_size, filter_condition=None, group_list=None):
    sample_type, target_dataset = None, None
    sample_set = set()
    if group_list:
        sample_type = 'stratified'
    else:
        sample_type = 'uniform'

    with open(cwd + '/workload/' + table_name + '/' + table_name + '.json', "r") as f:
        sample_json = json.load(f)
    categorical_attr_list = [col['field'] for col in sample_json['tables']['fact']['fields'] if
                             col['type'] == 'categorical']
    quantitative_attr_list = [col['field'] for col in sample_json['tables']['fact']['fields'] if
                              col['type'] == 'quantitative']

    if filter_condition:
        target_dataset = toolkit.query_parser.filter_parser(table_name, filter_condition)
        if len(target_dataset) > 1:
            for i in range(1, len(target_dataset) + 1):
                filter_comb = itertools.combinations(target_dataset.keys(), i)
                temp_filter_list = filter_condition.replace('(', '').replace(')', '').split()
                for f_c in filter_comb:
                    temp_name, temp_range, temp_filter = '', {}, ''
                    for j in range(len(f_c)):
                        f = f_c[j]
                        temp_range[f] = target_dataset[f]
                        for k in target_dataset[f]:
                            temp_name += f + k
                        t = 0
                        for k in range(len(temp_filter_list)):
                            if temp_filter_list[k] == f:
                                t += 1
                                if t > 1:
                                    num = 2
                                    while True:
                                        if k + num < len(temp_filter_list) and temp_filter_list[k + num] not in target_dataset.keys():
                                            if k + num + 1 < len(temp_filter_list):
                                                if temp_filter_list[k + num + 1] not in categorical_attr_list and temp_filter_list[k + num + 1] not in quantitative_attr_list:
                                                    num += 1
                                                else:
                                                    break
                                            else:
                                                num += 1
                                        else:
                                            break
                                    temp_filter += ' '.join(temp_filter_list[k - 1:k + num]) + ' '
                                else:
                                    num = 2
                                    while True:
                                        if k + num < len(temp_filter_list) and temp_filter_list[k + num] not in target_dataset.keys():
                                            if k + num + 1 < len(temp_filter_list):
                                                if temp_filter_list[k + num + 1] not in categorical_attr_list and temp_filter_list[k + num + 1] not in quantitative_attr_list:
                                                    num += 1
                                                else:
                                                    break
                                            else:
                                                num += 1
                                        else:
                                            break
                                    temp_filter += ' '.join(temp_filter_list[k:k + num]) + ' '
                        if len(f_c) - 1 > j:
                            temp_filter += 'and '
                    sample_name = table_name + '_' + sample_type + '_' + str(int(sample_size)) + '_' \
                                  + hashlib.md5(''.join(c for c in temp_name if c.isalnum()).encode(encoding='UTF-8')).hexdigest().replace('-', 'n')
                    sample_name = sample_name[0:63].lower()
                    sample = Sample(sample_name, sample_size, table_name, sample_type,
                                                     temp_range, temp_filter.strip(), group_list)
                    sample_set.add(sample)
        else:
            sample_name = table_name + '_' + sample_type + '_' + str(int(sample_size)) + '_' \
                          + hashlib.md5(
                ''.join(c for c in filter_condition if c.isalnum()).encode(encoding='UTF-8')).hexdigest().replace('-',
                                                                                                                  'n')
            sample_name = sample_name[0:63].lower()
            sample = Sample(sample_name, sample_size, table_name, sample_type,
                                             target_dataset, filter_condition, group_list)
            sample_set.add(sample)
    else:
        sample_name = table_name + '_' + sample_type + '_' + str(int(sample_size))
        sample_name = sample_name[0:63].lower()
        sample = Sample(sample_name, sample_size, table_name, sample_type, target_dataset, filter_condition, group_list)
        sample_set.add(sample)

    return sample_set


# construct sample according to a given query
def construct_query_2_samples(query, sample_size):
    table_name, filter_condition, group_list, _ = toolkit.query_parser.parser(query)

    global_sample = construct_sample(table_name, sample_size).pop()
    query_to_sample_dict[query] = [global_sample]

    if global_sample.sample_name not in sample_name_set:
        sample_name_set.add(global_sample.sample_name)
        sample_list.append(global_sample)
    else:
        for sample in sample_list:
            if sample.sample_name == global_sample.sample_name:
                global_sample.sample_cost = sample.sample_cost
                break

    if filter_condition:
        fine_sample_set = construct_sample(table_name, sample_size, filter_condition, group_list)
        for fine_sample in fine_sample_set:
            query_to_sample_dict[query].append(fine_sample)

            if fine_sample.sample_name not in sample_name_set:
                sample_name_set.add(fine_sample.sample_name)
                sample_list.append(fine_sample)
            else:
                for sample in sample_list:
                    if sample.sample_name == fine_sample.sample_name:
                        fine_sample.sample_cost = sample.sample_cost
                        break


# construct sample according to the given workload
def construct_workload_2_samples(test_workload, train_workload=[]):
    for query in test_workload:
        table_name, _, _, _ = toolkit.query_parser.parser(query)
        sample_size = toolkit.query_execute.underlying_size(table_name) * toolkit.tools.get_sampling_rate()
        if query not in train_workload:
            construct_query_2_samples(query, sample_size)
    materialize_samples(sample_list)


# materialize sample
def materialize(sample):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    cur.execute('Drop table if exists ' + sample.sample_name)
    if sample.sample_type == 'uniform':
        ratio = 0
        if sample.filter_condition:
            print('materialize: ', sample.filter_condition)
            sql = 'SELECT count(*) FROM ' + sample.target_table + ' WHERE ' + sample.filter_condition
            cur.execute(sql)
            data_size = int(cur.fetchall()[0][0])
            if data_size > 0:
                ratio = sample.sample_size / data_size
            ratio = 1 if ratio > 1 else ratio
            sql = 'Explain Analyze SELECT * into ' + sample.sample_name + ' FROM ' + sample.target_table + \
                  ' WHERE ' + sample.filter_condition + ' and random() <= ' + str(ratio)
        else:
            underlying_size = toolkit.query_execute.underlying_size(sample.target_table)

            if underlying_size > 0:
                ratio = sample.sample_size / underlying_size
            ratio = 1 if ratio > 1 else ratio
            sql = 'Explain Analyze SELECT * into ' + sample.sample_name + ' FROM ' + sample.target_table + \
                  ' WHERE random() < ' + str(ratio)
        cur.execute(sql)
        sample.sample_cost = float(cur.fetchall()[-1][0].split()[-2])
        sample.materialization = True
        sql = 'INSERT INTO sampleratio VALUES (' + "'" + sample.sample_name + "'," + "'all'," + str(ratio) + ')'
        cur.execute(sql)
    elif sample.sample_type == 'stratified':
        if sample.filter_condition:
            sql = 'SELECT ' + ', '.join(sample.group_list) + ', COUNT(*) FROM ' + sample.target_table + ' WHERE ' + \
                  sample.filter_condition + ' GROUP BY ' + ', '.join(sample.group_list)
        else:
            sql = 'SELECT ' + ', '.join(sample.group_list) + ', COUNT(*) FROM ' + sample.target_table + ' GROUP BY ' + \
                  ', '.join(sample.group_list)
        cur.execute(sql)
        group_count = cur.fetchall()
        group_size = sample.sample_size / len(group_count)
        group_count_dict = {}
        sum_ratio = 0
        for g_c in group_count:
            ratio = group_size / float(g_c[len(sample.group_list)])
            if ratio > 1:
                ratio = 1
            group_count_dict[g_c[:len(sample.group_list)]] = ratio
            sum_ratio += ratio
        cost = 0
        for i in range(len(group_count_dict.keys())):
            key = list(group_count_dict.keys())[i]
            ratio = group_count_dict[key]
            group = ''
            for j in range(len(key)):
                if j < len(sample.group_list):
                    group += sample.group_list[j] + ' = ' + "'" + key[j].strip() + "'" + ' and '
            if i == 0:
                if sample.filter_condition:
                    sql = 'Explain Analyze SELECT * into ' + sample.sample_name + ' FROM ' + sample.target_table + \
                          ' WHERE ' + group + sample.filter_condition + ' and random() < ' + str(ratio)
                else:
                    sql = 'Explain Analyze SELECT * into ' + sample.sample_name + ' FROM ' + sample.target_table + \
                          ' WHERE ' + group + 'random() < ' + str(ratio)
            else:
                if sample.filter_condition:
                    sql = 'Explain Analyze INSERT into ' + sample.sample_name + ' SELECT * FROM ' + sample.target_table + \
                          ' WHERE ' + group + sample.filter_condition + ' and random() <= ' + str(ratio)
                else:
                    sql = 'Explain Analyze INSERT into ' + sample.sample_name + ' SELECT * FROM ' + sample.target_table + \
                          ' WHERE ' + group + 'random() <= ' + str(ratio)
            cur.execute(sql)
            cost += float(cur.fetchall()[-1][0].split()[-2])
        sample.sample_cost = cost
        sample.materialization = True
        sql = 'INSERT INTO sampleratio VALUES (' + "'" + sample.sample_name + "'," + "'" + ','.join(
            sample.group_list) + "'," + str(sum_ratio / len(group_count_dict)) + ')'
        cur.execute(sql)

    conn.commit()
    conn.close()
    return sample.sample_cost


# materialize samples
def materialize_samples(candidate_sample_list):
    start = time.time()
    file = open(toolkit.tools.get_samples_name(), 'a+')
    materialized_samples_dict = toolkit.tools.get_samples()
    for sample in candidate_sample_list:
        if sample.sample_name not in materialized_samples_dict.keys():
            sample_cost = materialize(sample)
            file.write(sample.sample_name + ' ' + str(sample_cost) + '\n')
        else:
            sample.sample_cost = float(materialized_samples_dict[sample.sample_name])
            sample.materialization = True
    tc_ms = time.time() - start
    print('The time cost of materialize samples (s): ', tc_ms)


# delete materialized sample
def un_materialize_sample(sample):
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    sql = 'Drop table ' + sample.sample_name
    cur.execute(sql)
    conn.commit()
    conn.close()
