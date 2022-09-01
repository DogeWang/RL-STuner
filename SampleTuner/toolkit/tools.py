import json
import random
import csv
import os


cwd = os.getcwd()


def get_db_name():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    db_name = config_json['meta']['dbname']['name']
    return db_name


def get_k():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    k = config_json['meta']['k']['scale']
    return k


def get_storage_sample_number():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    number = config_json['meta']['storage_sample_number']['number']
    return number


def get_workload_file():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    filename = config_json['meta']['workload']['filename']
    return filename


def get_train_query_number():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    number = config_json['meta']['train_query_number']['number']
    return number


def get_error_threshold():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    error_threshold = config_json['meta']['error_threshold']['threshold']
    return error_threshold


def get_sampling_rate():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    sampling_rate = config_json['meta']['sampling_rate']['rate']
    return sampling_rate


def get_window_length():
    with open(cwd + '/config/config.json', "r") as f:
        config_json = json.load(f)
    n = config_json['meta']['window_length']['n']
    return n


def get_query_latency_result():
    query_data_dict = {}
    file_name = cwd + '/workload/' + get_db_name() + '/query_latency_result.csv'
    if os.path.isfile(file_name):
        csvfile = open(file_name, 'r')
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[-1].replace('.', '', 1).isdigit():
                query = row[1].replace(' WITH 1000', '')
                # query_data_dict[row[1]] = tuple([float(row[2]), float(row[-1])])
                query_data_dict[query] = tuple([float(row[2]), float(row[-1])])
    return query_data_dict


def get_query_latency_result_name():
    file_name = cwd + '/workload/' + get_db_name() + '/query_latency_result.csv'
    return file_name


def get_query_sample():
    query_sample_dict = {}
    file_name = cwd + '/workload/' + get_db_name() + '/query_sample_result.csv'
    if os.path.isfile(file_name):
        csvfile = open(file_name, 'r')
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[-1].replace('.', '', 1).isdigit():
                query, sample_name, latency, result = row[1].replace(' WITH 1000', ''), row[2], float(row[3]), float(row[-1])
                temp_tuple = tuple([query, sample_name])
                query_sample_dict[temp_tuple] = tuple([latency, result])
    return query_sample_dict


def get_query_sample_name():
    file_name = cwd + '/workload/' + get_db_name() + '/query_sample_result.csv'
    return file_name


def get_sample_statistic():
    sample_statistic_dict = {}
    file_name = cwd + '/workload/' + get_db_name() + '/query_sample_statistic.csv'
    if os.path.isfile(file_name):
        csvfile = open(file_name, 'r')
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[-1].replace('.', '', 1).isdigit():
                query, sample_name, count, stddev = row[1].replace(' WITH 1000', ''), row[2], float(row[3]), float(row[-1])
                temp_tuple = tuple([query, sample_name])
                sample_statistic_dict[temp_tuple] = tuple([count, stddev])
    return sample_statistic_dict


def get_sample_statistic_name():
    file_name = cwd + '/workload/' + get_db_name() + '/query_sample_statistic.csv'
    return file_name


def get_query_size():
    query_size_dict = {}
    file_name = cwd + '/workload/' + get_db_name() + '/query_size.csv'
    if os.path.isfile(file_name):
        csvfile = open(file_name, 'r')
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[-1].replace('.', '', 1).isdigit():
                query, size = row[1].replace(' WITH 1000', ''), int(row[2])
                query_size_dict[query] = size
    return query_size_dict


def get_query_size_name():
    file_name = cwd + '/workload/' + get_db_name() + '/query_size.csv'
    return file_name


def get_samples():
    materialized_samples_dict = {}
    file_name = cwd + '/workload/' + get_db_name() + '/materialized_samples.txt'
    if os.path.isfile(file_name):
        file = open(file_name, 'r')
        for row in file.readlines():
            sample_name, sample_cost = row.split(' ')[0], row.split(' ')[1].strip('\n')
            materialized_samples_dict[sample_name] = sample_cost
    return materialized_samples_dict


def get_samples_name():
    file_name = cwd + '/workload/' + get_db_name() + '/materialized_samples.txt'
    return file_name


def get_filter_condition():
    filter_condition_list = list()
    file_name = cwd + '/workload/' + get_db_name() + '/filter_condition.txt'
    if os.path.isfile(file_name):
        file = open(file_name, 'r')
        for row in file.readlines():
            filter_condition = row.strip('\n')
            filter_condition_list.append(filter_condition)
    return filter_condition_list


def get_filter_condition_name():
    file_name = cwd + '/workload/' + get_db_name() + '/filter_condition.txt'
    return file_name
