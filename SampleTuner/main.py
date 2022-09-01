import sys
import os
sys.path.append('/')

import toolkit.tools
import query_test


def experiment_basic(method, test_workload):
    dynamic_avg_l, dynamic_avg_e = 0, 0
    num = 1
    print('The performance of ' + method)
    for i in range(num):
        print()
        print('------------------------------------------------------ ', i,
              ' ------------------------------------------------------')
        print()
        avg_re_result_1 = query_test.test_basic(method, test_workload, i)
        dynamic_avg_e += avg_re_result_1 / num
    print('Final average relative error of ' + method + ' : ', dynamic_avg_e)


cwd = os.getcwd()
test_txt = open(cwd + '/workload/' + toolkit.tools.get_workload_file(), 'r')
test_workload = test_txt.readlines()
method_list = ['Taster', 'Taster-AW', 'Taster-FW', 'RL-STuner']

for method in method_list:
    experiment_basic(method, test_workload)





