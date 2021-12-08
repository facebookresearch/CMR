# bash experiments/run_scripts/slurm_er.sh qa val
# bash experiments/run_scripts/slurm_er.sh qa test

task=$1
mode=$2

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42 


if [ "$mode" = "val" ]; then
    echo "Validation mode for tuning hps"
    declare -a lrs=("1e-5" "3e-5" "5e-5")
    declare -a eps=("10" "20")
    declare -a l2ws=("0" "1" "10" "100")
    declare -a stream_ids=("0" "1" "2")
    declare -a rss=("32")
    declare -a rfs=("1" "3")
    for rs in "${rss[@]}"
    do
    for rf in "${rfs[@]}"
    do
    for lr in "${lrs[@]}"
    do
    for ep in "${eps[@]}"
    do
    for l2w in "${l2ws[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_er_ep=${ep}_lr=${lr}_l2w=${l2w}_${rs}_${rf}_si=${stream_id}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${ns_config} ${task} val ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
    done
    done
    done
    done

else
    echo "Testing mode with tuned hps"
    declare -a lrs=("3e-5")
    declare -a eps=("10")
    declare -a l2ws=("0")
    # declare -a stream_ids=("0" "1" "2" "3" "4") # 
    declare -a stream_ids=("5") # 
    declare -a rss=("32")
    declare -a rfs=("1" "3")
    for rs in "${rss[@]}"
    do
    for rf in "${rfs[@]}"
    do
    for lr in "${lrs[@]}"
    do
    for ep in "${eps[@]}"
    do
    for l2w in "${l2ws[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_er_ep=${ep}_lr=${lr}_l2w=${l2w}_${rs}_${rf}_si=${stream_id}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=120 --cpus-per-task 4 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${ns_config} ${task} test ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
    done
    done
    done
    done
fi



# declare -a lrs=("3e-5")
# declare -a eps=("10")
# declare -a rss=("32")
# declare -a rfs=("1" "3")
# declare -a l2ws=("0" "1" "10")
# for lr in "${lrs[@]}"
# do
# for ep in "${eps[@]}"
# do
# for rs in "${rss[@]}"
# do
# for rf in "${rfs[@]}"
# do
# for l2w in "${l2ws[@]}"
# do
# session_name=v1_er_${ep}_${lr}_${l2w}_${rs}_${rf}
# tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${ns_config} qa"
# echo "Created tmux session: ${session_name}"
# done
# done
# done
# done
# done

