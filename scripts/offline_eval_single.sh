# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

prefix=$1
thread=$2
n_threads=$3

log_file=bug_data/output/${prefix}_offline_eval/${prefix}_${thread}_of_${n_threads}.log
gpu=$(($start_gpuid+$thread))
CUDA_VISIBLE_DEVICES=0 /private/home/yuchenlin/.conda/envs/bartqa/bin/python semanticdebugger/debug_algs/run_lifelong_finetune.py \
    --prefix ${prefix} \
    --num_threads_eval ${n_threads} --current_thread_id ${thread} --max_timecode 50 \
    --path_to_thread_result bug_data/output/${prefix}_offline_eval/thread_${thread}_of_${n_threads}_result.json \
    --predict_batch_size 80 \
    --overtime_ckpt_dir bug_data/output/${prefix}_ckpts/ > ${log_file} 2>&1 &
tail -f ${log_file}