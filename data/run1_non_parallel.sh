#!/bin/bash

python3 Main_cc.py --config=./config/8987.json
python3 Main_cc.py --config=./config/batch_size_60000.json
python3 Main_cc.py --config=./config/no_weigh_decay.json
python3 Main_cc.py --config=./config/no_momentum.json
python3 Main_cc.py --config=./config/dropout_05.json