
declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
for seed in "${seeds[@]}"
do
session_name=index_nomir_-1_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 10 --pty exp_results/dynamic_stream/index_based/run_bart_index.sh 100 32 3 most_similar no 0 -1 ${seed}"
echo "Created tmux session: ${session_name}"

session_name=index_nomir_05_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 10 --pty exp_results/dynamic_stream/index_based/run_bart_index.sh 100 32 3 most_similar no 0 0.5 ${seed}"
echo "Created tmux session: ${session_name}"

session_name=index_mir_-1_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 10 --pty exp_results/dynamic_stream/index_based/run_bart_index.sh 100 32 3 most_similar yes 256 -1 ${seed}"
echo "Created tmux session: ${session_name}"

session_name=index_mir_05_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 10 --pty exp_results/dynamic_stream/index_based/run_bart_index.sh 100 32 3 most_similar yes 256 0.5 ${seed}"
echo "Created tmux session: ${session_name}"
done


