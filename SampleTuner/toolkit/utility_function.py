import toolkit.query_rewriter
import toolkit.tools
import toolkit.query_execute


alpha = 0.9
query_data_dict = toolkit.tools.get_query_latency_result()
query_sample_dict = toolkit.tools.get_query_sample()
sample_statistic_dict = toolkit.tools.get_sample_statistic()
app_query_sample_result_dict = {}
app_query_sample_benefit_dict = {}
error_threshold = 0
workload_length = 0


def get_accuracy_reward(relative_error):
    accuracy_reward = alpha * -relative_error
    if relative_error < error_threshold:
        accuracy_reward = alpha * (1 - error_threshold)
    return accuracy_reward


def exact_relative_error(query, sample):
    temp_tuple = tuple([query, sample.sample_name])

    if query.find('GROUP BY') > 0:
        if temp_tuple in query_sample_dict.keys():
            _, sample_result = query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
        else:
            _, sample_result = toolkit.query_execute.exact_group_query_latency_result(query, sample)

        if query in query_data_dict.keys():
            dataset_latency, dataset_result = query_data_dict[query][0], query_data_dict[query][1]
        else:
            dataset_latency, dataset_result = toolkit.query_execute.exact_group_query_latency_result(query)

        sum_re = 0
        if dataset_result:
            for key in dataset_result.keys():
                if key in sample_result.keys():
                    sum_re += abs((sample_result[key] - dataset_result[key]) / dataset_result[key])
                else:
                    sum_re += 1
            relative_error = sum_re / len(dataset_result)
        else:
            relative_error = 0
    else:
        if temp_tuple in query_sample_dict.keys():
            _, sample_result = query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
        else:
            _, sample_result = toolkit.query_execute.exact_query_latency_result(query, sample)

        if query in query_data_dict.keys():
            dataset_latency, dataset_result = query_data_dict[query][0], query_data_dict[query][1]
        else:
            dataset_latency, dataset_result = toolkit.query_execute.exact_query_latency_result(query)

        if dataset_result != 0:
            relative_error = abs((sample_result - dataset_result) / dataset_result)
        else:
            relative_error = 0

    return relative_error


def exact_sample_benefit(query, sample):
    temp_tuple = tuple([query, sample.sample_name])
    if temp_tuple in query_sample_dict.keys():
        sample_latency, _ = query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
    else:
        if query.find('GROUP BY') > 0:
            sample_latency, _ = toolkit.query_execute.exact_group_query_latency_result(query, sample)
        else:
            sample_latency, _ = toolkit.query_execute.exact_query_latency_result(query, sample)

    if query in query_data_dict.keys():
        dataset_latency, dataset_result = query_data_dict[query][0], query_data_dict[query][1]
    else:
        if query.find('GROUP BY') > 0:
            dataset_latency, dataset_result = toolkit.query_execute.exact_group_query_latency_result(query)
        else:
            dataset_latency, dataset_result = toolkit.query_execute.exact_query_latency_result(query)

    if not dataset_result:
        accuracy_reward = 0
    else:
        relative_error = exact_relative_error(query, sample)
        accuracy_reward = get_accuracy_reward(relative_error)

    latency_reward = (1 - alpha) * ((dataset_latency - sample_latency) / dataset_latency)
    benefit = accuracy_reward + latency_reward

    return benefit


def map_query_sample_benefit(workload, materialized_samples, exact=True):
    query_sample, sample_query = {}, {}
    total_benefit = 0
    for sample in materialized_samples:
        sample_query[sample] = []
    for query in workload:
        samples = toolkit.query_rewriter.choose_candidate_active_sample(query, materialized_samples)
        if len(samples) >= 1:
            max_sample, max_benefit = None, -1000000
            for sample in samples:
                if exact:
                    benefit = exact_sample_benefit(query, sample)
                if max_benefit < benefit:
                    max_sample = sample
                    max_benefit = benefit
            total_benefit += max_benefit
            query_sample[query] = max_sample
            sample_query[max_sample].append(query)
        else:
            query_sample[query] = None

    return query_sample, sample_query, total_benefit


def exact_new_sample_utility(workload, sample):
    n_s = len(workload)
    utility = 0

    for query in workload:
        temp_tuple = tuple([query, sample.sample_name])

        if temp_tuple in query_sample_dict.keys():
            sample_latency, _ = query_sample_dict[temp_tuple][0], query_sample_dict[temp_tuple][1]
        else:
            if query.find('GROUP BY') > 0:
                sample_latency, _ = toolkit.query_execute.exact_group_query_latency_result(query, sample)
            else:
                sample_latency, _ = toolkit.query_execute.exact_query_latency_result(query, sample)

        if query in query_data_dict.keys():
            dataset_latency, dataset_result = query_data_dict[query][0], query_data_dict[query][1]
        else:
            if query.find('GROUP BY') > 0:
                dataset_latency, dataset_result = toolkit.query_execute.exact_group_query_latency_result(query)
            else:
                dataset_latency, dataset_result = toolkit.query_execute.exact_query_latency_result(query)

        if not dataset_result:
            accuracy_reward = 0
        else:
            relative_error = exact_relative_error(query, sample)
            accuracy_reward = get_accuracy_reward(relative_error)
        if not sample.sample_cost:
            print(sample.sample_name, sample.sample_cost)
        latency_reward = (1 - alpha) * ((dataset_latency - sample_latency - sample.sample_cost / n_s) / dataset_latency)
        utility += accuracy_reward + latency_reward
    return utility


def calculate_action_utility(workload, materialized_samples, exact=True):
    query_sample, sample_query, _ = map_query_sample_benefit(workload, materialized_samples, exact)
    global workload_length
    if workload_length <= 50:
        p = 0
    else:
        p = -1

    utility = 0
    for m_sample in materialized_samples:
        for sample in sample_query.keys():
            if m_sample.equal(sample):
                if exact:
                    utility += exact_new_sample_utility(sample_query[sample], sample)
                break

    for query in query_sample.keys():
        if not query_sample[query]:
            utility += p

    return utility, sample_query, query_sample


def calculate_sample_benefit(sample_query, exact=True):
    sample_benefit_dict = {}
    for sample in sample_query.keys():
        queries = sample_query[sample]
        benefit = 0
        if queries:
            for query in queries:
                if exact:
                    benefit += exact_sample_benefit(query, sample)
        sample_benefit_dict[sample] = benefit
    return sample_benefit_dict


def calculate_sample_utility(sample_query, exact=True):
    sample_utility_dict = {}
    query_re_dict = {}
    for sample in sample_query.keys():
        queries = sample_query[sample]
        if exact:
            utility = exact_new_sample_utility(queries, sample)
            for query in queries:
                relative_error = exact_relative_error(query, sample)
                query_re_dict[query] = relative_error
        if not queries:
            utility = -1
        sample_utility_dict[sample] = utility

    return sample_utility_dict, query_re_dict


def map_sample_utility(workload, sample, exact=True):
    query_list = []
    for query in workload:
        if toolkit.query_rewriter.choose_candidate_active_sample(query, [sample]):
            query_list.append(query)
    utility = exact_new_sample_utility(query_list, sample) + len(query_list)
    return utility
