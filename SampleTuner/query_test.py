import psycopg2
import pandas as pd
import toolkit.tools
import toolkit.query_execute
import toolkit.error_estimator
import toolkit.query_parser
import toolkit.utility_function
import core.taster
import core.rl_stuner
import toolkit.sample_materialize
import toolkit.query_rewriter


db_name = toolkit.tools.get_db_name()
query_data_dict = toolkit.tools.get_query_latency_result()
query_sample_dict = toolkit.tools.get_query_sample()
query_size_dict = toolkit.tools.get_query_size()
sample_statistic_dict = toolkit.tools.get_sample_statistic()
train_number = toolkit.tools.get_train_query_number()
query_selectivity_dict = {}
error_threshold = toolkit.tools.get_error_threshold()


def test(method, test_workload, file_name, active_sample_set=set()):
    global query_data_dict, query_selectivity_dict
    query_latency_list, relative_error_list, query_selectivity_list, usable_bool_list, query_size_list = [], [], [], [], []
    conn = psycopg2.connect(database=db_name, user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    i, sum_relative_error, sample_tuning_number, last_invokation_num = 1, 0, 0, 0

    for query in test_workload:
        chosen_active_sample = toolkit.query_rewriter.map_query_sample_execute(query, active_sample_set)
        if chosen_active_sample:
            print('Query number:', i + train_number, chosen_active_sample.sample_name, chosen_active_sample.target_dataset, chosen_active_sample.filter_condition)
        else:
            print('Query number:', i + train_number)

        # get query selectivity
        table_name, _, _, _ = toolkit.query_parser.parser(query)
        total_data_size = toolkit.query_execute.underlying_size(table_name)
        if query in query_selectivity_dict.keys():
            query_size = query_selectivity_dict[query]
        else:
            temp_list = query[:query.find(';')].split(' ')
            del temp_list[1:temp_list.index('FROM')]
            temp_list.insert(1, 'COUNT(*)')
            query_count = ' '.join(temp_list)
            cur.execute(query_count)
            query_size = int(cur.fetchall()[0][0])
            query_selectivity_dict[query] = query_size / total_data_size
        query_selectivity_list.append(query_size / total_data_size)

        if chosen_active_sample:
            if query.find('GROUP BY') > 0:
                relative_error = toolkit.utility_function.exact_relative_error(query, chosen_active_sample)
            else:
                # read exact query result from .csv, reduce experiment time
                if query in query_data_dict.keys():
                    _, underlying_result = query_data_dict[query][0], query_data_dict[query][1]
                else:
                    underlying_latency, underlying_result = toolkit.query_execute.exact_query_latency_result(query)
                    query_data_dict[query] = tuple([underlying_latency, underlying_result])

                # read approximate query result from .csv, reduce experiment time
                temp_tuple = tuple([query, chosen_active_sample.sample_name])
                if temp_tuple in query_sample_dict:
                    sample_latency, sample_result = query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
                else:
                    sample_latency, sample_result = toolkit.query_execute.exact_query_latency_result(query,
                                                                                                     chosen_active_sample)

                if underlying_result == 0:
                    relative_error = 0
                else:
                    relative_error = abs((sample_result - underlying_result) / underlying_result)

            relative_error_list.append(relative_error)
            query_latency_list.append(sample_latency)

            # sample tuning
            if relative_error <= error_threshold:
                usable_bool_list.append(1)
            else:
                sample_tuning_number += 1
                print('The relative error is larger than the error threshold.')
                if True:
                    if method == 'Taster':
                        active_sample_set = core.taster.sample_tuning(test_workload[:i])
                    elif method == 'Taster-AW':
                        active_sample_set = core.taster.sample_tuning_aw(test_workload[:i], test_workload[last_invokation_num:i])
                    elif method == 'Taster-FW':
                        active_sample_set = core.taster.sample_tuning_fw(test_workload[:i])
                    elif method == 'RL-STuner':
                        active_sample_set = core.rl_stuner.lazy_sample_tuning(test_workload[:i],
                                                                                     active_sample_set, method)
                usable_bool_list.append(-2)
                last_invokation_num = i
        else:
            sample_tuning_number += 1
            relative_error_list.append(1)
            query_latency_list.append(-1)
            usable_bool_list.append(-1)
            last_invokation_num = i

            print('Query cannot be executed on materialized samples -1: ', query)
            if True:
                if method == 'Taster':
                    active_sample_set = core.taster.sample_tuning(test_workload[:i])
                elif method == 'Taster-AW':
                    active_sample_set = core.taster.sample_tuning_aw(test_workload[:i],
                                                                     test_workload[last_invokation_num:i])
                elif method == 'Taster-FW':
                    active_sample_set = core.taster.sample_tuning_fw(test_workload[:i])
                elif method == 'RL-STuner':
                    active_sample_set = core.rl_stuner.lazy_sample_tuning(test_workload[:i],
                                                                          active_sample_set, method)
        i += 1

    avg_relative_error = sum(relative_error_list) / len(test_workload)
    print('The number of sample tuning: ', sample_tuning_number)
    print('Average Relative error: ', avg_relative_error)
    conn.commit()
    conn.close()
    toolkit.sample_materialize.sample_list.clear()
    toolkit.sample_materialize.sample_name_set.clear()

    dataframe = pd.DataFrame({'Query Latency': query_latency_list, 'Relative Error': relative_error_list, 'Selectivity': query_selectivity_list, 'Usable': usable_bool_list})
    dataframe.to_csv(file_name, sep=',', mode='a')

    return avg_relative_error


def test_basic(method, test_workload, num=0):
    global query_data_dict
    query_data_dict.update(toolkit.query_execute.query_execute_data(test_workload))

    avg_re_result = test(method, test_workload, 'statistical_of_dynamic_workload_' + method + '_' + str(num) + '.csv')
    return avg_re_result