# RL-STuner
Learning-based Sample Tuning Method for Approximate Query Processing in Interactive Data Exploration

## Setup:

Linux

PostgreSQL 11.2

Python 3.6+ and Pytorch 1.6

## Workload and Config Files:

/config/config.json

change workload, storage budget, sampling ratio, and so on.

/workload/flights/

short session workload, middle session workload, long session workload, exploratory workload, and flights.json (attributes).

/workload/tpch/

exploratory workload and lineitem.json (attributes).

## Running Sample Tuner:

nohup python3 -u main.py &
