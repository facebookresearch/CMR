
# declare -a seeds=("")
for seed in {1..32}
do
session_name=sim_dm_${seed}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=100 --cpus-per-task 4 --pty exp_results/supervision_data/run_simple_mir_dm.sh ${seed} 1"
echo "Created tmux session: ${session_name}"
done

