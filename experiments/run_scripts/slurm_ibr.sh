
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42
# declare -a lrs=("3e-5" "5e-5")
# declare -a eps=("3" "5" "10")
declare -a lrs=("3e-5")
declare -a eps=("10")
declare -a rss=("32" "64")
declare -a rfs=("1" "3")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for rs in "${rss[@]}"
do
for rf in "${rfs[@]}"
do
session_name=v1_ibr_${ep}_${lr}_${rs}_${rf}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_ibr.sh ${lr} ${ep} ${rs} ${rf} 0.5 ${ns_config} qa"
echo "Created tmux session: ${session_name}"
done
done
done
done

