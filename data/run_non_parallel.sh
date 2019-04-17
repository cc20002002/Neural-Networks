#!/bin/bash

python3 Main_cc.py --config=./config/8987.json
python3 Main_cc.py --config=./config/batch_size_60000.json
python3 Main_cc.py --config=./config/no_weigh_decay.json
python3 Main_cc.py --config=./config/no_momentum.json
python3 Main_cc.py --config=./config/dropout_05.json

python3 Main_cc.py --config=./config/8987.json
python3 Main_cc.py --config=./config/8995.json
python3 Main_cc.py --config=./config/8972.json
python3 Main_cc.py --config=./config/8975.json
python3 Main_cc.py --config=./config/8932.json