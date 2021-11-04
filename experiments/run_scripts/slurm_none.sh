ns_config="T=100,b=64,alpha=0.98,beta=1,gamma=1"
session_name=noencl
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=90 --cpus-per-task 10 --pty experiments/run_scripts/run_nonecl.sh ${ns_config}"
echo "Created tmux session: ${session_name}"


