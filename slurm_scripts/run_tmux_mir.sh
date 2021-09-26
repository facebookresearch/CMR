
declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
for seed in "${seeds[@]}"
do
session_name=mir-c4096_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 4096 none ${seed}"
echo "Created tmux session: ${session_name}"
done

