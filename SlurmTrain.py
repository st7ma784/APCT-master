from test_tube import SlurmCluster
#from trainclip_v2 import train as train_clip
from trainclip_v2 import testtrainfunc as train_clip


from HOparser import parser
if __name__ == '__main__':
    from functools import partial
    train=partial(train_clip,dir="/nobackup/projects/bdlan05/smander3/data/")


    argsparser = parser(strategy='random_search')
    hyperparams = argsparser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path="/nobackup/projects/bdlan05/smander3/logs/",#hyperparams.log_path,
        python_cmd='python3',
#        test_tube_exp_name="PL_test"
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(
        email='st7ma784@gmail.com', on_done=True, on_fail=True)

    # SLURM Module to load.
    # cluster.load_modules([
    #     'python-3',
    #     'anaconda3'
    # ])

    # Add commands to the non-SLURM portion.
    
    cluster.add_command('source activate open-ce') # We'll assume that on the BEDE/HEC cluster you've named you conda env after the standard...

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    cluster.add_slurm_cmd(
        cmd='account', value='bdlan05', comment='Project account for Bede')
    cluster.add_slurm_cmd(
        cmd='partition', value='gpu', comment='request gpu partition on Bede')

    # Set job compute details (this will apply PER set of hyperparameters.)
    #print(cluster.__dir__())
    #del cluster.memory_mb_per_node
    #This is commented because on bede, having gone into 
    #nano /nobackup/projects/bdlan05/smander3/miniconda/envs/open-ce/lib/python3.9/site-packages/test_tube/hpc.py
    #and removed memory per node and adjusted to not include cpu counts as this is done automatically in bede 
    #del cluster.per_experiment_nb_cpus
    cluster.cpus_per_task=0
    cluster.per_experiment_nb_gpus = 4
    cluster.per_experiment_nb_nodes = 2
    #cluster.gpu_type = '1080ti'

    # we'll request 100GB of memory per node
    #cluster.memory_mb_per_node = 0

    # set a walltime of 24 hours,0, minues
    cluster.job_time = '24:00:00'

    # 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
    # you must provide your own loading and saving function which the cluster object will call
    cluster.minutes_to_checkpoint_before_walltime = 1
    print(cluster.__dir__())
    # run the models on the cluster
    cluster.optimize_parallel_cluster_gpu(train, nb_trials=2, job_name='fourth_wandb_trial_batch') # Change this to optimize_parralel_cluster_cpu to debug.
