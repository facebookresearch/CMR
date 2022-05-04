
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ns_config="T=100,b=128,alpha=0.9,beta=0.5,gamma=0.8"
seed=42

 
ns_name=$(echo "${ns_config}" | tr '.' '#')
ns_name=$(echo "${ns_name}" | tr ',' '#')

echo $ns_name


# NoneCL baseline 
session_name=${ns_name}_nli_noencl
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=125 --cpus-per-task 4 --pty experiments/run_scripts/run_nonecl.sh ${ns_config} nli no"
echo "Created tmux session: ${session_name}"

# CFT 
declare -a lrs=("3.5e-5")
declare -a eps=("10")
declare -a l2ws=("0")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for l2w in "${l2ws[@]}"
do
session_name=${ns_name}_nli_cft
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_simplecl.sh ${seed} ${lr} ${ep} ${l2w} ${ns_config} nli"
echo "Created tmux session: ${session_name}"
done
done
done

# OnlineEWC

declare -a lrs=("3.5e-5")
declare -a eps=("10")
declare -a lambdas=("250")
declare -a gammas=("9e-1")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for lambda in "${lambdas[@]}"
do
for gamma in "${gammas[@]}"
do
session_name=${ns_name}_nli_oewc
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_oewc.sh ${seed} ${lr} ${ep} ${lambda} ${gamma} ${ns_config} nli"
echo "Created tmux session: ${session_name}"
done
done
done
done

# ER

declare -a lrs=("3.5e-5")
declare -a eps=("10")
declare -a rss=("32")
declare -a rfs=("1" "3")
declare -a l2ws=("0")
for lr in "${lrs[@]}"
do
for ep in "${eps[@]}"
do
for rs in "${rss[@]}"
do
for rf in "${rfs[@]}"
do
for l2w in "${l2ws[@]}"
do
session_name=${ns_name}_nli_er_rf=${rf}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_er.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${ns_config} nli"
echo "Created tmux session: ${session_name}"
done
done
done
done
done


# MIR

declare -a lrs=("3.5e-5")
declare -a eps=("10")
declare -a rss=("32")
declare -a rfs=("1" "3") 
declare -a mir_cand_sizes=("256")
declare -a mir_configs=("none")
declare -a l2ws=("0" "1")
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
for mconfg in "${mir_configs[@]}"
do
for l2w in "${l2ws[@]}"
do
session_name=${ns_name}_nli_mir_rf=${rf}_l2w=${l2w}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --constraint=volta32gb --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_mir.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${mcs} ${mconfg} ${ns_config} nli "
echo "Created tmux session: ${session_name}"

done
done
done
done
done
done
done


# MaxLoss

declare -a lrs=("3.5e-5")
declare -a eps=("10")
declare -a rss=("32")
declare -a rfs=("1" "3") 
declare -a mir_cand_sizes=("256")
declare -a mir_configs=("largest_afterloss")
declare -a l2ws=("0")
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
for mconfg in "${mir_configs[@]}"
do
for l2w in "${l2ws[@]}"
do
session_name=${ns_name}_nli_maxloss_rf=${rf}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --constraint=volta32gb --partition=devlab --time=180 --cpus-per-task 4 --pty experiments/run_scripts/run_mir.sh ${lr} ${ep} ${l2w} ${rs} ${rf} 0.5 ${mcs} ${mconfg} ${ns_config} nli "
echo "Created tmux session: ${session_name}"

done
done
done
done
done
done
done 