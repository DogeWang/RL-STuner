import scipy.stats
import math
import toolkit.tools

confidence_level = 0.95
bootstrap_num = 100
db_name = toolkit.tools.get_db_name()


def avg_closed_form(sample_size, underlying_size, stddev):
    if sample_size > 0:
        z = scipy.stats.norm.ppf(confidence_level)
        error_bound = z * math.sqrt(1 - sample_size / underlying_size) * stddev / math.sqrt(sample_size)
    else:
        error_bound = 1000000
    return error_bound


