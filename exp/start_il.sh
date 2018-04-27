#!/bin/bash

model=perfect.caffemodel
port=5560

echo "Generating data with $model"
echo "Subscribed on $port"

sh +x kill.sh
rm log/*.rc? log/*.hfo
nohup python hfo_env.py train $model 12700 $port > run00.log 2>/dev/null &
nohup python hfo_env.py train $model 12710 $port > run01.log 2>/dev/null &
nohup python hfo_env.py train $model 12720 $port > run02.log 2>/dev/null &
nohup python hfo_env.py train $model 12730 $port > run03.log 2>/dev/null &
nohup python hfo_env.py train $model 12740 $port > run04.log 2>/dev/null &
nohup python hfo_env.py train $model 12750 $port > run05.log 2>/dev/null &
nohup python hfo_env.py train $model 12760 $port > run06.log 2>/dev/null &
nohup python hfo_env.py train $model 12770 $port > run07.log 2>/dev/null &
nohup python hfo_env.py train $model 12780 $port > run08.log 2>/dev/null &
nohup python hfo_env.py train $model 12790 $port > run09.log 2>/dev/null &
nohup python hfo_env.py train $model 12800 $port > run10.log 2>/dev/null &
nohup python hfo_env.py train $model 12810 $port > run11.log 2>/dev/null &
nohup python hfo_env.py train $model 12820 $port > run12.log 2>/dev/null &
nohup python hfo_env.py train $model 12830 $port > run13.log 2>/dev/null &
nohup python hfo_env.py train $model 12840 $port > run14.log 2>/dev/null &
nohup python hfo_env.py train $model 12850 $port > run15.log 2>/dev/null &
nohup python hfo_env.py train $model 12860 $port > run16.log 2>/dev/null &
nohup python hfo_env.py train $model 12870 $port > run17.log 2>/dev/null &
nohup python hfo_env.py train $model 12880 $port > run18.log 2>/dev/null &
nohup python hfo_env.py train $model 12890 $port > run19.log 2>/dev/null &
#nohup python hfo_env.py test init.caffemodel 12900 $port > test.log 2>/dev/null &
