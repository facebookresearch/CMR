
# declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
# for seed in "${seeds[@]}"
# do
# session_name=ER_T100_F3_${seed}
# tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=100 --cpus-per-task 10 --pty exp_results/dynamic_stream/memory_based/run_er.sh 100 32 3 0.5 ${seed}"
# echo "Created tmux session: ${session_name}"
# done



declare -a seeds=("42" "0212" "1213" "2021" "123" "456" "567" "789")
for seed in "${seeds[@]}"
do
session_name=er_05_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_er.sh 100 32 3 0.5 ${seed}"
echo "Created tmux session: ${session_name}"


session_name=er_02_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_er.sh 100 32 3 0.2 ${seed}"
echo "Created tmux session: ${session_name}"


session_name=er_08_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_er.sh 100 32 3 0.8 ${seed}"
echo "Created tmux session: ${session_name}"

session_name=er_-1_T100_F3_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 8 --pty exp_results/dynamic_stream/memory_based/run_mir.sh 100 32 3 -1 ${seed}"
echo "Created tmux session: ${session_name}"
done