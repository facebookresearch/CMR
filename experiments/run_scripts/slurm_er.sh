seed=42
declare -a lrs=("1e-5")
declare -a eps=("3")
declare -a rss=("32")
declare -a rfs=("1")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for rs in "${rss[@]}"
do
for rf in "${rfs[@]}"
do
session_name=er_${ep}_${lr}_${rs}_${rf}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=90 --cpus-per-task 10 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${rs} ${rf} 0.5"
echo "Created tmux session: ${session_name}"
done
done
done
done

