import submitit
function = submitit.helpers.CommandFunction(["which", "python"])
executor = submitit.SlurmExecutor(folder="log_test")
executor.update_parameters(time=1, partition="learnfair", gpus_per_node=1)
job = executor.submit(function)

# The returned python path is the one used in slurm.
# It should be the same as when running out of slurm!
# This means that everything that is installed in your
# conda environment should work just as well in the cluster
print(job.result())