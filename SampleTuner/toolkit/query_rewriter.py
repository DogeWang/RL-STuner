import json
import os
import toolkit.tools
import toolkit.query_parser
import toolkit.query_execute


cwd = os.getcwd()


# return True if query.colmun >= sample.colmun and query.target_dataset <= sample.target_dataset
# i.e., query can be executed on the given sample
def judge_query_sample_filter(table_name, query_filter_dict, sample_filter_dict):
    if not (set(query_filter_dict.keys()) >= set(sample_filter_dict.keys())):
        return False

    with open(cwd + '/workload/' + table_name + '/' + table_name + '.json', "r") as f:
        sample_json = json.load(f)
    categorical_attr_list = [col['field'] for col in sample_json['tables']['fact']['fields'] if
                              col['type'] == 'categorical']
    quantitative_attr_list = [col['field'] for col in sample_json['tables']['fact']['fields'] if
                              col['type'] == 'quantitative']

    for key in sample_filter_dict.keys():
        if key in categorical_attr_list:
            if not (set(query_filter_dict[key]) <= set(sample_filter_dict[key])):
                return False
        elif key in quantitative_attr_list:
            sample_filter_list = sample_filter_dict[key]
            query_filter_list = query_filter_dict[key]
            for i in range(len(query_filter_list)):
                judge = False
                x_0, x_1 = float(query_filter_list[i].split('~')[0]), float(query_filter_list[i].split('~')[1])
                for j in range(len(sample_filter_list)):
                    y_0, y_1 = float(sample_filter_list[j].split('~')[0]), float(sample_filter_list[j].split('~')[1])
                    if x_0 >= y_0 and x_1 <= y_1:
                        judge = True
                        break
                if not judge:
                    return False
    return True


def choose_candidate_active_sample(query, active_sample_set):
    candidate_active_sample_set = set()
    table_name, filter_condition, group_list, _ = toolkit.query_parser.parser(query)

    for sample in active_sample_set:
        if sample.target_table != table_name:
            continue
        if group_list and sample.sample_type == 'stratified' and len(group_list) > len(sample.group_list):
            continue
        if group_list and sample.sample_type == 'stratified' and not (set(group_list) <= set(sample.group_list)):
            continue
        if filter_condition and sample.target_dataset:
            query_data_range = toolkit.query_parser.filter_parser(table_name, filter_condition)
            if not judge_query_sample_filter(table_name, query_data_range, sample.target_dataset):
                continue
        if not filter_condition and sample.target_dataset:
            continue
        candidate_active_sample_set.add(sample)

    return candidate_active_sample_set


def map_workload_sample_execut(test_workload, active_sample_set):
    query_sample, sample_query = {}, {}
    for sample in active_sample_set:
        sample_query[sample] = []
    for query in test_workload:
        candidate_samples = choose_candidate_active_sample(query, active_sample_set)
        if len(candidate_samples) >= 1:
            max_ratio = 0
            best_sample = None
            for sample in candidate_samples:
                ratio = toolkit.query_execute.sample_ratio(sample.sample_name)
                if max_ratio < ratio:
                    max_ratio = ratio
                    best_sample = sample
                if max_ratio >= 1:
                    break
            query_sample[query] = best_sample
            if best_sample:
                sample_query[best_sample].append(query)
        else:
            query_sample[query] = None
    return sample_query, query_sample


def map_query_sample_execute(query, active_sample_set):
    chosen_active_sample = None
    candidate_active_sample_set = choose_candidate_active_sample(query, active_sample_set)

    if len(candidate_active_sample_set) >= 1:
        max_ratio = 0
        best_sample = None
        for sample in candidate_active_sample_set:
            ratio = toolkit.query_execute.sample_ratio(sample.sample_name)
            if max_ratio < ratio:
                max_ratio = ratio
                best_sample = sample
            if max_ratio >= 1:
                break
        chosen_active_sample = best_sample

    return chosen_active_sample


def map_query_sample_taster(test_workload, sample_list):
    sample_query, query_sample = {}, {}

    for sample in sample_list:
        sample_query[sample] = []

    for query in test_workload:
        query_sample[query] = []
        candidate_samples = choose_candidate_active_sample(query, sample_list)
        if len(candidate_samples) >= 1:
            for sample in candidate_samples:
                sample_query[sample].append(query)
                query_sample[query].append(sample)

    return sample_query, query_sample
