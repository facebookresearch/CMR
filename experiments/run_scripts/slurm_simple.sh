ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42

declare -a lrs=("3e-5" "5e-5")
declare -a eps=("5" "10" "20")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
session_name=v3_simple_ep=${ep}_lr=${lr}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_simplecl.sh ${seed} ${lr} ${ep} ${ns_config}"
echo "Created tmux session: ${session_name}"
done
done

