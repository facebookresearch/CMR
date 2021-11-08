ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42
declare -a lrs=("1e-5" "3e-5") 
declare -a eps=("3" "6" "9")
declare -a rss=("32" "64")
declare -a rfs=("1" "3")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for rs in "${rss[@]}"
do
for rf in "${rfs[@]}"
do
session_name=v3_er_${ep}_${lr}_${rs}_${rf}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=125 --cpus-per-task 10 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${rs} ${rf} 0.5 ${ns_config}"
echo "Created tmux session: ${session_name}"
done
done
done
done

