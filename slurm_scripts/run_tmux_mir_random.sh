
declare -a seeds=("42" "10212" "11213" "12021" "1123" "1456" "1567" "1789")
for seed in "${seeds[@]}"
do
session_name=mir-random_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 10 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 32 random ${seed}"
echo "Created tmux session: ${session_name}"
done

