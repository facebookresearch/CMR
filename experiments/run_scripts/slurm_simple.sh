# bash experiments/run_scripts/slurm_simple.sh qa val

task=$1
mode=$2

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42

if [ "$mode" = "val" ]; then
    echo "Validation mode for tuning hps"
    declare -a lrs=("3e-5" "1e-5" "5e-5")
    declare -a eps=("10" "20")
    declare -a l2ws=("0" "1" "10" "100")
    declare -a stream_ids=("0" "1" "2")
    for lr in "${lrs[@]}"
    do
    for ep in "${eps[@]}"
    do
    for l2w in "${l2ws[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_simple_ep=${ep}_lr=${lr}_l2w=${l2w}_si=${stream_id}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=60 --cpus-per-task 4 --pty experiments/run_scripts/run_simplecl.sh ${seed} ${lr} ${ep} ${l2w} ${ns_config} ${task} val ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
    done
    done

else
    echo "test mode"
fi
