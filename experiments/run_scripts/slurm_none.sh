# bash experiments/run_scripts/slurm_none.sh qa val
# bash experiments/run_scripts/slurm_none.sh qa test

task=$1
mode=$2

ns_config="T=100,b=64,alpha=0.9,beta=0.5,gamma=0.8"
seed=42 


if [ "$mode" = "val" ]; then 
    declare -a stream_ids=("0" "1" "2")
else
    # declare -a stream_ids=("0" "1" "2" "3" "4") # 
    declare -a stream_ids=("5")
fi

for stream_id in "${stream_ids[@]}"
do
    session_name=noencl_mode=${mode}_si=${stream_id}
    tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=125 --cpus-per-task 4 --pty experiments/run_scripts/run_nonecl.sh ${ns_config} ${task} no ${mode} ${stream_id}"
    echo "Created tmux session: ${session_name}"
done 
