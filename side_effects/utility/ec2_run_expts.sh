#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

configs=`ls ./configs`
echo $configs
for i in $configs; do session=`cut -d"." -f1 <<< "$i"`; echo $session; tmux new-session -s $session -d;  tmux send-keys -t $session:0 "conda activate deeplib ; python ./expts_routines.py -p ./configs/$i \;" C-m;done
tmux ls
