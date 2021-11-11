ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42

# declare -a lrs=("3e-5" "5e-5")
# declare -a eps=("5" "10")
# declare -a lambdas=("500" "250" "1000")
declare -a lrs=("3e-5")
declare -a eps=("10")
declare -a lambdas=("1" "10" "100" "500" "250" "1000")
declare -a gammas=("1" "-1" "9e-1")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for lambda in "${lambdas[@]}"
do
for gamma in "${gammas[@]}"
do
session_name=v1_oewc_ep=${ep}_lr=${lr}_lbd=${lambda}_gm=${gamma}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_oewc.sh ${seed} ${lr} ${ep} ${lambda} ${gamma} ${ns_config} qa"
echo "Created tmux session: ${session_name}"
done
done
done
done

