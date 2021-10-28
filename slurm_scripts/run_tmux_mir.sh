
declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
for seed in "${seeds[@]}"
do
session_name=mir_05_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 256 none 0.5 ${seed}"
echo "Created tmux session: ${session_name}"


session_name=mir_02_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 256 none 0.2 ${seed}"
echo "Created tmux session: ${session_name}"


session_name=mir_08_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 256 none 0.8 ${seed}"
echo "Created tmux session: ${session_name}"

session_name=mir_-1_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 256 none -1 ${seed}"
echo "Created tmux session: ${session_name}"
done

