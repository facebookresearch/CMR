# declare -a hidden_dims=("256") # "256")
# declare -a dim_vectors=("128") #"128")
# declare -a lrs=("1e-3" "1e-4") # "1e-4") #
# declare -a droprates=("0" "0.3") #
# declare -a qry_sizes=("16") # "16") #
# declare -a pos_sizes=("8" "16")  #
# declare -a neg_sizes=("1" "8" "16") # "8" ) #
# declare -a batch_sizes=("128") # "128"
# declare -a use_query_means=("False") # "True" 

# declare -a hidden_dims=("512") # "256")
# declare -a dim_vectors=("256") #"128")
# declare -a lrs=("1e-3" "1e-4") # "1e-4") #
# declare -a droprates=("0" "0.3") #
# declare -a qry_sizes=("16") # "16") #
# declare -a pos_sizes=("8")  #
# declare -a neg_sizes=("8" "16") # "8" ) #
# declare -a batch_sizes=("128") # "128"
# declare -a use_query_means=("False") # "True" 

declare -a hidden_dims=("768" "1024") # "256")
declare -a dim_vectors=("384") #"128")
declare -a lrs=("1e-3" "1e-4") # "1e-4") #
declare -a droprates=("0.3" "0.5") #
declare -a qry_sizes=("8") # "16") #
declare -a pos_sizes=("8")  #
declare -a neg_sizes=("16") # "8" ) #
declare -a batch_sizes=("128" "64") # "128"
declare -a use_query_means=("False") # "True" 



# cp -r exp_results/supervision_data/1020_dm_simple/ /tmp/
# ds_dir_path="/tmp/1020_dm_simple/"
ds_dir_path="exp_results/supervision_data/1020_dm_simple"
index=0 
n_gpus=2
for hidden_dim in "${hidden_dims[@]}"
do
for dim_vector in "${dim_vectors[@]}"
do
for lr in "${lrs[@]}"
do
for droprate in "${droprates[@]}"
do
for qry_size in "${qry_sizes[@]}"
do
for pos_size in "${pos_sizes[@]}"
do
for neg_size in "${neg_sizes[@]}"
do
for batch_size in "${batch_sizes[@]}"
do
for use_query_mean in "${use_query_means[@]}"
do

session_name="${hidden_dim}_${dim_vector}_${lr}_${droprate}_${qry_size}_${pos_size}_${neg_size}_${batch_size}"
# tmux new-session -d -s ${session_name} "
log_file="logs/tmp/${session_name}.log"
touch $log_file
gpu=$((index % n_gpus))
CUDA_VISIBLE_DEVICES=$gpu python semanticdebugger/debug_algs/index_based/biencoder.py \
    --ds_dir_path ${ds_dir_path} \
    --use_cuda True --wandb True  --save_ckpt False --use_query_mean ${use_query_mean} --hidden_dim ${hidden_dim} --dim_vector ${dim_vector} --lr ${lr} --batch_size ${batch_size} --droprate ${droprate} --qry_size ${qry_size} --pos_size ${pos_size} --neg_size ${neg_size} > ${log_file} &
sleep 5
index=$((index+1))
echo "Index:${index} GPU:$gpu Log: ${log_file}"

done
done
done
done
done
done
done
done
done
