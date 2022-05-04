
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
session_name=v1_noencl_offline
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=125 --cpus-per-task 10 --pty experiments/run_scripts/run_nonecl.sh ${ns_config} qa yes"
echo "Created tmux session: ${session_name}"


