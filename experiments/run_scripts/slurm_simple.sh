seed=42
ns_config="T=100,b=64,alpha=0.98,beta=1,gamma=1"
declare -a lrs=("1e-5" "3e-5" "5e-5")
declare -a eps=("1" "3" "6")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
session_name=simple_ep=${ep}_lr=${lr}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=90 --cpus-per-task 10 --pty experiments/run_scripts/run_simplecl.sh ${seed} ${lr} ${ep} ${ns_config}"
echo "Created tmux session: ${session_name}"
done
done

