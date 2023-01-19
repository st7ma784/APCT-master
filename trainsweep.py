import wandb
wandb.login()
if __name__=="__main__":
    sweep_config = {
        'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[2e-4,1e-4,5e-5,2e-5]
            },
            'batch_size': {
                'values': [8,12,16,24,32]
            },
            'precision': {
                'values': ['bf16']
            },
            'embed_dim':{
                'values': [128,256,512]
            }, 
            'transformer_width':{
                'values': [128,256,512]
            },
            'codeversion':{
                'values':[1,3,4,5,6]
            'transformer_heads':{
                'values': [8,16,32]
            },
            'transformer_layers':{
                'values': [4,5,6]
            },
        }
    }
'''        self.opt_list("--learning_rate", default=0.00001, type=float, options=[1e-3,1e-5, 1e-4,], tunable=True)
        self.opt_list("--batch_size", default=10, type=int, options=[6,8,10,12], tunable=True)
        self.opt_list("--JSE", default=0, type=int, options=[0], tunable=True)
        self.opt_list("--precision", default=16, options=[16], tunable=False)
        self.opt_list("--codeversion", default=6, type=int, options=[6], tunable=False)
        self.opt_list("--transformer_layers", default=5, type=int, options=[3,4,5,6], tunable=True)
        self.opt_list("--transformer_heads", default=16, type=int, options=[16], tunable=True)
        self.opt_list("--embed_dim", default=512, type=int, options=[128,512], tunable=True)
        self.opt_list("--transformer_width", default=512, type=int, options=[128,512], tunable=True)
        self.opt_list("--devices", default=1, type=int, options=[1], tunable=False)
        self.opt_list("--accelerator", default='gpu', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)'''
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="6DimCachespliteinSweep", entity="st7ma784")
    print(sweep_id)
