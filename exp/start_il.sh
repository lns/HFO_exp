#!/bin/bash

model=init.caffemodel
port=5560

echo "Generating data with $model"
echo "Subscribed on $port"

sh +x kill.sh
rm log/*.rc? log/*.hfo
nohup python hfo_env.py train $model 12700 $port 0 IL $1 > run00.log 2>train.err &
nohup python hfo_env.py train $model 12710 $port 0 IL $1 > run01.log 2>/dev/null &
nohup python hfo_env.py train $model 12720 $port 0 IL $1 > run02.log 2>/dev/null &
nohup python hfo_env.py train $model 12730 $port 0 IL $1 > run03.log 2>/dev/null &
nohup python hfo_env.py train $model 12740 $port 0 IL $1 > run04.log 2>/dev/null &
nohup python hfo_env.py train $model 12750 $port 0 IL $1 > run05.log 2>/dev/null &
nohup python hfo_env.py train $model 12760 $port 0 IL $1 > run06.log 2>/dev/null &
nohup python hfo_env.py train $model 12770 $port 0 IL $1 > run07.log 2>/dev/null &
nohup python hfo_env.py train $model 12780 $port 0 IL $1 > run08.log 2>/dev/null &
nohup python hfo_env.py train $model 12790 $port 0 IL $1 > run09.log 2>/dev/null &
nohup python hfo_env.py train $model 12800 $port 0 IL $1 > run10.log 2>/dev/null &
nohup python hfo_env.py train $model 12810 $port 0 IL $1 > run11.log 2>/dev/null &
nohup python hfo_env.py train $model 12820 $port 0 IL $1 > run12.log 2>/dev/null &
nohup python hfo_env.py train $model 12830 $port 0 IL $1 > run13.log 2>/dev/null &
nohup python hfo_env.py train $model 12840 $port 0 IL $1 > run14.log 2>/dev/null &
nohup python hfo_env.py train $model 12850 $port 0 IL $1 > run15.log 2>/dev/null &
nohup python hfo_env.py train $model 12860 $port 0 IL $1 > run16.log 2>/dev/null &
nohup python hfo_env.py train $model 12870 $port 0 IL $1 > run17.log 2>/dev/null &
nohup python hfo_env.py train $model 12880 $port 0 IL $1 > run18.log 2>/dev/null &
nohup python hfo_env.py train $model 12890 $port 0 IL $1 > run19.log 2>/dev/null &
nohup python hfo_env.py test init.caffemodel 12900 $port 1 IL $1 > test.log 2>test.err &
