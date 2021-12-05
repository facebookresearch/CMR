# bash experiments/run_scripts/slurm_oewc.sh qa val
# bash experiments/run_scripts/slurm_oewc.sh qa test

task=$1
mode=$2

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42


if [ "$mode" = "val" ]; then
    echo "Validation mode for tuning hps"
    declare -a lrs=("1e-5" "3e-5" "5e-5")
    declare -a eps=("10" "20")
    declare -a lambdas=("1" "10" "100" "500" "250" "1000")
    declare -a gammas=("1" "-1" "9e-1")
    declare -a stream_ids=("0" "1" "2")
    for lr in "${lrs[@]}"
    do
    for ep in "${eps[@]}"
    do
    for lambda in "${lambdas[@]}"
    do
    for gamma in "${gammas[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_oewc_ep=${ep}_lr=${lr}_lbd=${lambda}_gm=${gamma}_si=${stream_id}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=120 --cpus-per-task 4 --pty experiments/run_scripts/run_oewc.sh ${seed} ${lr} ${ep} ${lambda} ${gamma} ${ns_config} ${task} val ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
    done
    done
    done

else
    echo "Testing mode with tuned hps"
    declare -a lrs=("3e-5")
    declare -a eps=("10")
    declare -a lambdas=("500" "100" "1000")
    declare -a gammas=("1")
    declare -a stream_ids=("0" "1" "2" "3" "4")
    for lr in "${lrs[@]}"
    do
    for ep in "${eps[@]}"
    do
    for lambda in "${lambdas[@]}"
    do
    for gamma in "${gammas[@]}"
    do
    for stream_id in "${stream_ids[@]}"
    do
    session_name=${task}_oewc_ep=${ep}_lr=${lr}_lbd=${lambda}_gm=${gamma}_si=${stream_id}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=120 --cpus-per-task 4 --pty experiments/run_scripts/run_oewc.sh ${seed} ${lr} ${ep} ${lambda} ${gamma} ${ns_config} ${task} test ${stream_id}"
    echo "Created tmux session: ${session_name}"
    done
    done
    done
    done
    done
fi




# declare -a lrs=("3e-5")
# declare -a eps=("10")
# declare -a lambdas=("1" "10" "100" "500" "250" "1000")
# declare -a gammas=("1" "-1" "9e-1")
# for lr in "${lrs[@]}"
# do
# for ep in "${eps[@]}"
# do
# for lambda in "${lambdas[@]}"
# do
# for gamma in "${gammas[@]}"
# do
# session_name=v1_oewc_ep=${ep}_lr=${lr}_lbd=${lambda}_gm=${gamma}
# tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 10 --pty experiments/run_scripts/run_oewc.sh ${seed} ${lr} ${ep} ${lambda} ${gamma} ${ns_config} qa"
# echo "Created tmux session: ${session_name}"
# done
# done
# done
# done

