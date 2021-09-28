
declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
for seed in "${seeds[@]}"
do
session_name=sim_ep=7_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=100 --cpus-per-task 10 --pty exp_results/dynamic_stream/cl_simple/run_simple.sh ${seed} 7"
echo "Created tmux session: ${session_name}"
done
