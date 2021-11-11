ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42

declare -a lrs=("3e-5")
declare -a eps=("10")
declare -a l2ws=("0" "1" "10" "100")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for l2w in "${l2ws[@]}"
do
session_name=v1_simple_ep=${ep}_lr=${lr}_l2w=${l2w}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_simplecl.sh ${seed} ${lr} ${ep} ${l2w} ${ns_config} qa"
echo "Created tmux session: ${session_name}"
done
done
done

