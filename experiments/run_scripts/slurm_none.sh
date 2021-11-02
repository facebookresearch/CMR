session_name=noencl
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=90 --cpus-per-task 10 --pty experiments/run_scripts/run_nonecl.sh"
echo "Created tmux session: ${session_name}"


