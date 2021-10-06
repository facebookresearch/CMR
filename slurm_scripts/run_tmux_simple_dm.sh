
# declare -a seeds=("")
for seed in {1..500}
do
session_name=sim_dm_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=15 --cpus-per-task 10 --pty exp_results/supervision_data/run_simple_mir_dm.sh ${seed}"
echo "Created tmux session: ${session_name}"
done

