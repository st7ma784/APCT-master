
import os,sys

def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None,project="6DIMCLIPTOKSweepv4",entity="st7ma784"):
    import pytorch_lightning

    if config is not None:
        config=config.__dict__
        dir=config.get("dir",dir)
        logtool= pytorch_lightning.loggers.WandbLogger( project=project,entity=entity, save_dir=dir)

    else: 
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        import wandb
        print("here")
        run=wandb.init(project=project,entity=entity,name=project,config=config)
        logtool= pytorch_lightning.loggers.WandbLogger( project=project,entity=entity,experiment=run, save_dir=dir)
        config=run.config.as_dict()
    
    train(config,dir,devices,accelerator,Dataset,logtool)

def train(config={
        "batch_size":16,
        "learning_rate":2e-3,
        "precision":16,
        "embed_dim": 512,
        "codeversion":4,
        "transformer_width": 512,
        "transformer_heads": 32,
        "transformer_layers": 4,
        "JSE":False,
    },dir=None,devices=None,accelerator=None,Dataset=None,logtool=None):

    import pytorch_lightning
    version=int(config.get("codeversion",-1))
    
    from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
    from model.LossCalculation import calculate_loss,calculate_loss2,calculate_loss3,calculate_loss4,calculate_loss5,calculate_lossStock
    from model.PruneCalculation import PruneHook
    from model.trainclip_v53 import LightningCLIPModule
    
    # if version==22:
    #     from modelVersions.trainclip_v53_noproj import LightningCLIPModule
    # elif version==21:
    #     from modelVersions.trainclip_v53_projinv import LightningCLIPModule

    # elif version==20:
    #     from modelVersions.trainclip_v53_proji import LightningCLIPModule
    # elif version==18:
    #     from modelVersions.trainclip_v48_fxloss import LightningCLIPModule
    # elif version==17:
    #     from modelVersions.trainclip_v53_fxloss import LightningCLIPModule
    # elif version==16:
    #     from modelVersions.trainclip_v52_fxloss import LightningCLIPModule
    # elif version==15:
    #     from modelVersions.trainclip_v53 import LightningCLIPModule
    # elif version==14:
    #     from modelVersions.trainclip_v52 import LightningCLIPModule
    # elif version==13:
    #     from modelVersions.trainclip_v51 import LightningCLIPModule
    # elif version==12:
    #     from modelVersions.trainclip_v50 import LightningCLIPModule
    # elif version==11:
    #     from modelVersions.trainclip_v49 import LightningCLIPModule
    # elif version==10:
    #     from modelVersions.trainclip_v48_BaselineStock import LightningCLIPModule
    # elif version==9:
    #     from modelVersions.trainclip_v48_Baseline import LightningCLIPModule #this is for n=2 with my loss
    # elif version==8:
    #     from modelVersions.trainclip_v48_Entropy import LightningCLIPModule
    # elif version==7:
    #     from modelVersions.trainclip_v48_Prune import LightningCLIPModule
    # elif version==6:
    #     from modelVersions.trainclip_v48_var import LightningCLIPModule
    # elif version==5:
    #     from modelVersions.trainclip_v47_var import LightningCLIPModule
    # elif version==4:
    #     from modelVersions.trainclip_v46_var import LightningCLIPModule
    # elif version==3:
    #     from modelVersions.trainclip_v45_var import LightningCLIPModule
    # elif version==1:
    #     from modelVersions.trainclip_v37_einsumimp import LightningCLIPModule
    # else:
    #     print("CONFIG",config)
    model=LightningCLIPModule( train_batch_size=config["batch_size"],
                                **config)
    if dir is None:
        dir=config.get("dir",".")
    if Dataset is None:
        from BuildSpainDataSet import COCODataModule
        from BuildLAION import LaionDataModule
        
        #Dataset=LaionDataModule(Cache_dir=dir,batch_size=config["batch_size"])
        Dataset=COCODataModule(Cache_dir=dir,batch_size=config["batch_size"])
    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("accelerator","auto")
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE
    #for windows .... 
    if sys.platform == "win32":
       os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    print("Launching with precision",p)
    trainer=pytorch_lightning.Trainer(
            devices=devices,
            #auto_select_gpus=True,
            accelerator=accelerator,
            max_epochs=40,
            #profiler="advanced",
            logger=logtool,
            strategy="dp",
            num_nodes=int(os.getenv("SLURM_NNODES",1)),
            callbacks=callbacks,
            #gradient_clip_val=0.25,# Not supported for manual optimization
            fast_dev_run=False,
            precision=p
    )
    if config["batch_size"] !=1:
        
        trainer.fit(model,Dataset)
    else:
        return 0 #No need to train if batch size is 1

def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',   
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version), 
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',  #Tasks per node
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        #'#SBATCH --gres=gpu:{}:{}'.format(self.gpu_type, self.per_experiment_nb_gpus),    If you want to specify a GPU type
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('st7ma784@gmail.com'),
    ]
    comm="python"
    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
        '#SBATCH --account=bdlan05',
        'export CONDADIR=/nobackup/projects/bdlan05/$USER/miniconda',])
        #slurm_commands={"account":"bdlan05"}#,"partition":"gpu"} Leaving this part out to run on non-bede slurm
        comm="python3"
    else: 
        sub_commands.extend(['export CONDADIR=/home/user/miniconda3',])
        #slurm_commands={}
    #sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        '#SBATCH --mem-per-node=62G',  #Memory per node
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'source $CONDADIR/etc/profile.d/conda.sh',
        'conda activate open-ce',# ...and activate the conda environment
    ])
    #sub_commands.append("srun python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)

    sub_commands.append('{} {} {}'.format(comm, script_name,trialArgs))
    #when launched, this script will be called with no trials, and so drop into the wandbtrain section, 
    sub_commands = [x.lstrip() for x in sub_commands]        

    full_command = '\n'.join(sub_commands)
    return full_command

def __get_hopt_params(trial):
    """
    Turns hopt trial into script params
    :param trial:
    :return:
    """
    params = []
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v=0
        # don't add None params
        if v is None or v is False:
            continue

        # put everything in quotes except bools
        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    # this arg lets the hyperparameter optimizer do its thin
    full_cmd = ' '.join(params)
    return full_cmd

def __should_escape(v):
    v = str(v)
    return '[' in v or ';' in v or ' ' in v


if __name__ == '__main__':
    from HOparser import parser
    from subprocess import call
    myparser=parser()
    hyperparams = myparser.parse_args()
    defaultConfig=hyperparams.__dict__
   
    NumTrials=hyperparams.num_trials
    #BEDE has Env var containing hostname  #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node
    if NumTrials==-1:
        trial=hyperparams.generate_trials(1)[0]
        print("Running trial: {}".format(trial))
        wandbtrain(trial)

    elif NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        wandbtrain(hyperparams)

    #OR To run with Default Args
    else: 
        trials=hyperparams.generate_trials(NumTrials)

        for i,trial in enumerate(trials):             
            command=SlurmRun(trial)
            slurm_cmd_script_path = os.path.join(defaultConfig.get("dir","."),"slurm_cmdtrial{}.sh".format(i))
            with open(slurm_cmd_script_path, "w") as f:
              f.write(command)
            print('\nlaunching exp...')
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')  