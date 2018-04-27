ps -ef | grep "hfo_env" | awk '{print $2}' | xargs kill
ps -ef | grep "bin/HFO" | awk '{print $2}' | xargs kill
killall -9 rcssserver
