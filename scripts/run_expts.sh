#!/usr/bin/env bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

configs=`ls ./side_effects/configs`
echo $configs
for i in $configs; do session=`cut -d"." -f1 <<< "$i"`; echo $session; tmux new-session -s $session -d;  tmux send-keys -t $session:0 "conda activate deeplib ; python ./side_effects/expts_routines.py -p ./side_effects/configs/$i \;" C-m;done
tmux ls
