import copy
import random
import toolkit.sample_materialize
import toolkit.tools
import toolkit.query_rewriter
import toolkit.utility_function


n = 3
a_parameter = 0.25
storage_sample_number = toolkit.tools.get_storage_sample_number()


def taster(test_workload, sample_list, n=toolkit.tools.get_window_length(), train_workload=[]):
    print('n:', n)
    toolkit.utility_function.workload_length = len(test_workload + train_workload)
    if train_workload and test_workload == []:
        toolkit.utility_function.workload_length = len(train_workload)
    if len(test_workload) >= n:
        workload = test_workload[-n:]
    else:
        workload = test_workload[:]
    if train_workload:
        workload += train_workload

    active_samples_list, utility_list, utility_sample = [], [], {}
    for sample in sample_list:
        temp_utility, _, _ = toolkit.utility_function.calculate_action_utility(workload, [sample])
        utility_list.append((sample.sample_name, temp_utility))
        utility_sample[sample] = temp_utility

    if len(active_samples_list) < storage_sample_number:
        temp_list = sorted(utility_sample.items(), key=lambda item: item[1], reverse=True)
        active_samples_list.append(temp_list[0][0])
        del utility_sample[temp_list[0][0]]

    while len(active_samples_list) < storage_sample_number:
        if len(utility_sample) <= 0:
            break
        cur = {}
        for sample, _ in utility_sample.items():
            cur[sample.sample_name] = False
        while True:
            temp_list = sorted(utility_sample.items(), key=lambda item: item[1], reverse=True)
            temp_sample = temp_list[0][0]
            if cur[temp_sample.sample_name]:
                active_samples_list.append(temp_sample)
                del utility_sample[temp_sample]
                break
            else:
                temp_utility, _, _ = toolkit.utility_function.calculate_action_utility(workload,
                                                                                  active_samples_list + [temp_sample])
                utility_sample[temp_sample] = temp_utility - utility_sample[temp_sample]
                cur[temp_sample.sample_name] = True

    print('The materialized samples after tuning by Taster are: ', end='')
    for sample in active_samples_list:
        print(sample.sample_name, end=' ')
    print()
    return set(active_samples_list)


def sample_tuning(test_workload, train_workload=[]):
    print(
        '||==========================================================================================================================================================||')
    print(
        '||                                                                                                                                                          ||')
    print(
        '||                                                                               Taster                                                                     ||')
    print(
        '||                                                                                                                                                          ||')
    print(
        '||==========================================================================================================================================================||')

    toolkit.sample_materialize.construct_workload_2_samples(test_workload, train_workload)
    sample_list = copy.deepcopy(toolkit.sample_materialize.sample_list)
    active_sample_set = taster(test_workload, sample_list, train_workload=train_workload)

    print('The materialized samples after tuning by Taster are: ', end='')
    for sample in active_sample_set:
        print(sample.sample_name, end=' ')
    print()

    return active_sample_set


def sample_tuning_aw(test_workload, new_workload, train_workload=[]):
    print(
        '||==========================================================================================================================================================||')
    print(
        '||                                                                                                                                                          ||')
    print(
        '||                                                                              Taster-AW                                                                   ||')
    print(
        '||                                                                                                                                                          ||')
    print(
        '||==========================================================================================================================================================||')

    global n
    train_workload, test_workload = copy.deepcopy(train_workload), copy.deepcopy(test_workload)
    toolkit.sample_materialize.construct_workload_2_samples(test_workload, train_workload)
    sample_list = copy.deepcopy(toolkit.sample_materialize.sample_list)

    n_lower = int((1 - a_parameter) * n)
    if n_lower <= 0:
        n_lower = 1
    n_upper = int((1 + a_parameter) * n)
    active_sample_set_n = taster(test_workload, sample_list, n, train_workload=train_workload)
    active_sample_set_nlower = taster(test_workload, sample_list, n_lower, train_workload=train_workload)
    active_sample_set_nupper = taster(test_workload, sample_list, n_upper, train_workload=train_workload)

    benefit_n, _, _ = toolkit.utility_function.calculate_action_utility(new_workload, active_sample_set_n, True)
    benefit_nlower, _, _ = toolkit.utility_function.calculate_action_utility(new_workload, active_sample_set_nlower, True)
    benefit_nupper, _, _ = toolkit.utility_function.calculate_action_utility(new_workload, active_sample_set_nupper, True)

    active_sample_set = set()
    if benefit_n >= benefit_nupper and benefit_n >= benefit_nlower:
        n = n
        active_sample_set = active_sample_set_n
    elif benefit_nupper > benefit_n and benefit_nupper > benefit_nlower:
        n = n_upper
        active_sample_set = active_sample_set_nupper
    elif benefit_nlower > benefit_n and benefit_nlower > benefit_nupper:
        n = n_lower
        active_sample_set = active_sample_set_nlower

    print('The window length of n is: ', n)
    print('The materialized samples after tuning by Taster-AW are: ', end='')
    for sample in active_sample_set:
        print(sample.sample_name, end=' ')
    print()

    return active_sample_set


def sample_tuning_fw(test_workload, train_workload=[]):
    print(
        '||==========================================================================================================================================================||')
    print(
        '||                                                                                                                                                          ||')
    print(
        '||                                                                              Taster-FW                                                                   ||')
    print(
        '||                                                                                                                                                          ||')
    print(
        '||==========================================================================================================================================================||')

    toolkit.sample_materialize.construct_workload_2_samples(test_workload, train_workload)
    sample_list = copy.deepcopy(toolkit.sample_materialize.sample_list)
    active_sample_set = taster(test_workload, sample_list, n=150, train_workload=train_workload)

    print('The materialized samples after tuning by Taster-FW are: ', end='')
    for sample in active_sample_set:
        print(sample.sample_name, end=' ')
    print()

    return active_sample_set
