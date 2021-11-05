ns_config="T=100,b=64,alpha=0.98,beta=1,gamma=1"
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
session_name=v1_er_${ep}_${lr}_${rs}_${rf}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=120 --cpus-per-task 10 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${rs} ${rf} 0.5 ${ns_config}"
echo "Created tmux session: ${session_name}"
done
done
done
done

