# RL-STuner
RL-STuner: A Deep Reinforcement Learning-based Sample Tuning Method for Approximate Query Processing

## Setup:

Linux

PostgreSQL 11.2

Python 3.6 and Pytorch 1.6

## Workload and Config Files:

/config/config.json

change workload, storage budget, sampling rate, and so on.

/workload/flights/

short session workload, middle session workload, long session workload, exploratory workload, and flights.json (attributes).

/workload/tpch/

exploratory workload and lineitem.json (attributes).

/workload/example_workload.txt
example workload

## Running Sample Tuner:

nohup python3 -u main.py &
