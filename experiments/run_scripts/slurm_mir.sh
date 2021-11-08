ns_config="T=100,b=64,alpha=0.9,beta=0.1,gamma=0.8"
seed=42

# declare -a lrs=("3e-5" "5e-5")
# declare -a eps=("3" "5" "10")
declare -a lrs=("3e-5")
declare -a eps=("10")
declare -a rss=("32" "64")
declare -a rfs=("1" "3")
# declare -a mir_cand_sizes=("256" "512")
declare -a mir_cand_sizes=("256")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for rs in "${rss[@]}"
do
for rf in "${rfs[@]}"
do
for mcs in "${mir_cand_sizes[@]}"
do
session_name=v1_mir_${ep}_${lr}_${rs}_${rf}_${mcs}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_mir.sh ${lr} ${ep} ${rs} ${rf} 0.5 ${mcs} ${ns_config}"
echo "Created tmux session: ${session_name}"
done
done
done
done
done

