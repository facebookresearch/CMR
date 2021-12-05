# bash experiments/run_scripts/slurm_mir.sh qa val
# bash experiments/run_scripts/slurm_mir.sh qa test

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
    declare -a mir_cand_sizes=("256")
    declare -a mir_configs=("none")

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
    for mcs in "${mir_cand_sizes[@]}"
    do
    for mconfg in "${mir_configs[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_mir_ep=${ep}_lr=${lr}_l2w=${l2w}_${rs}_${rf}_si=${stream_id}_${mcs}_${mconfg}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_mir.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${mcs} ${mconfg} ${ns_config} ${task} val ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
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
    declare -a stream_ids=("0" "1" "2" "3" "4") # 
    declare -a rss=("32")
    declare -a rfs=("1" "3")
    declare -a mir_cand_sizes=("256" "512")
    declare -a mir_configs=("none" "largest_afterloss")

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
    for mcs in "${mir_cand_sizes[@]}"
    do
    for mconfg in "${mir_configs[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_mir_ep=${ep}_lr=${lr}_l2w=${l2w}_${rs}_${rf}_si=${stream_id}_${mcs}_${mconfg}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=120 --cpus-per-task 4 --pty experiments/run_scripts/run_mir.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${mcs} ${mconfg} ${ns_config} ${task} test ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
    done
    done
    done
    done
    done
    done
fi


 



# # declare -a lrs=("3e-5" "5e-5")
# # declare -a eps=("3" "5" "10")
# declare -a lrs=("3e-5")
# declare -a eps=("10")
# declare -a rss=("32")
# declare -a rfs=("1" "3") 
# declare -a mir_cand_sizes=("256")
# declare -a mir_configs=("none")
# declare -a l2ws=("1" "5" "10")
# for lr in "${lrs[@]}"
# do
# for ep in "${eps[@]}"
# do
# for rs in "${rss[@]}"
# do
# for rf in "${rfs[@]}"
# do
# for mcs in "${mir_cand_sizes[@]}"
# do
# for mconfg in "${mir_configs[@]}"
# do
# for l2w in "${l2ws[@]}"
# do
# session_name=v2_mir_${ep}_${lr}_${rs}_${rf}_${mcs}_${mconfg}_${l2w}
# tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --constraint=volta32gb --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_mir.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${mcs} ${mconfg} ${ns_config} qa "
# echo "Created tmux session: ${session_name}"

# done
# done
# done
# done
# done
# done
# done

