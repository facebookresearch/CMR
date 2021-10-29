
declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
for seed in "${seeds[@]}"
do
session_name=IO_sim25_index_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=100 --cpus-per-task 10 --pty exp_results/dynamic_stream/index_based/run_bart_io_index.sh 100 32 3 most_similar no 0 0.5 ${seed}"
echo "Created tmux session: ${session_name}"
done
