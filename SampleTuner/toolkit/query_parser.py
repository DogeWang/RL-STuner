import json
import os


cwd = os.getcwd()
tpch_sample_table = 'lineitem'


def parser(query):
    table_name, filter_condition, group_by, group_list = None, None, None, None

    if query.find('WHERE') > 0:
        table_name = query[query.find('FROM') + len('FROM'): query.find('WHERE')].strip()
        if query.find('GROUP BY') > 0:
            filter_condition = query[query.find('WHERE') + len('WHERE'): query.find('GROUP BY')].strip()
            group_by = query[query.find('GROUP BY') + len('GROUP BY'): query.find(';')].strip()
        else:
            filter_condition = query[query.find('WHERE') + len('WHERE'): query.find(';')].strip()
    elif query.find('GROUP BY') > 0:
        table_name = query[query.find('FROM') + len('FROM'): query.find('GROUP BY')].strip()
        group_by = query[query.find('GROUP BY') + len('GROUP BY'): query.find(';')].strip()
    else:
        table_name = query[query.find('FROM') + len('FROM'): query.find(';')].strip()

    if group_by:
        group_list = group_by.split(',')
        for i in range(len(group_list)):
            group_list[i] = group_list[i].strip()

    join_condition = ''
    if ',' in table_name and tpch_sample_table in table_name:
        table_name = tpch_sample_table
        with open(cwd + '/workload/tpch/' + table_name + '.json', "r") as f:
            table_json = json.load(f)
    else:
        with open(cwd + '/workload/' + table_name + '/' + table_name + '.json', "r") as f:
            table_json = json.load(f)

    foreign_keys = table_json["tables"]["dimension"]["foreign_key"].split(',')
    temp_list = filter_condition.split('))')
    for temp in temp_list:
        for foreign in foreign_keys:
            if foreign.strip() in temp:
                join_condition += temp + ' and '
    join_condition = join_condition.replace('(', '').replace(')', '')
    filter_condition = filter_condition.replace('(', '').replace(')', '')
    filter_condition = filter_condition.replace(join_condition, '')

    return table_name, filter_condition, group_list, join_condition


def filter_parser(table_name, filter_condition):
    data_range = {}

    op_list = ['>=', '>', '<=', '<']
    with open(cwd + '/workload/' + table_name + '/' + table_name + '.json', "r") as f:
        sample_json = json.load(f)
    categorical_attr_list = [col['field'] for col in sample_json['tables']['fact']['fields'] if
                              col['type'] == 'categorical']
    quantitative_attr_list = [col['field'] for col in sample_json['tables']['fact']['fields'] if
                              col['type'] == 'quantitative']

    filter_list = filter_condition.split(')')
    for i in range(len(filter_list) - 1, -1, -1):
        if filter_list[i] == '':
            filter_list.remove('')
    for i in range(len(filter_list)):
        filter_list[i] = filter_list[i].replace('(', '').strip()
        factor_list = filter_list[i].split(' ')
        for j in range(len(factor_list)):
            if factor_list[j] in categorical_attr_list:
                attr_range = factor_list[j + 2].replace("'", '')
                k = j + 3
                while True:
                    if k < len(factor_list) and factor_list[k] not in categorical_attr_list and factor_list[k] not in quantitative_attr_list:
                        attr_range += ' ' + factor_list[k].replace("'", '')
                        k += 1
                    else:
                        break
                if factor_list[j] in data_range:
                    data_range[factor_list[j]].append(attr_range)
                else:
                    data_range[factor_list[j]] = [attr_range]
            elif factor_list[j] in quantitative_attr_list:
                if j > 0:
                    attr_range = factor_list[j - 1] + ' ' + ' '.join(factor_list[j + 1:j + 3])
                else:
                    attr_range = ' '.join(factor_list[j + 1:j + 3])
                if factor_list[j] in data_range:
                    data_range[factor_list[j]].append(attr_range)
                else:
                    data_range[factor_list[j]] = [attr_range]

    for key in data_range.keys():
        if key in quantitative_attr_list:
            transform_list = []
            temp_list = data_range[key]
            i = 0
            while True:
                if i >= len(temp_list):
                    break
                else:
                    if float(temp_list[i].split(' ')[-1]) <= float(temp_list[i+1].split(' ')[-1]):
                        transform_list.append(temp_list[i].split(' ')[-1] + '~' + temp_list[i + 1].split(' ')[-1])
                    else:
                        transform_list.append(temp_list[i + 1].split(' ')[-1] + '~' + temp_list[i].split(' ')[-1])
                i += 2

            data_range[key] = transform_list
    return data_range


def aggregation_parser(query):
    aggregation_list = []
    for temp in query.split(' '):
        if temp.lower().find('avg(') == 0 or temp.lower().find('sum(') == 0 or temp.lower().find('count(') == 0:
            aggregation_type = temp.lower().split('(')[0]
            aggregation_attribution = temp.lower().split('(')[-1].replace(')', '').replace(',','')
            aggregation_list.append(tuple([aggregation_type, aggregation_attribution]))

    return aggregation_list
