import wandb
wandb.login()
if __name__=="__main__":
    sweep_config = {
        'method': 'bayes',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
            },
            'batch_size': {
                'values': [1,4,8,16,32,64]
            },
            'useclip_im': {
                'values': [True,False]
            },
            'useclip_en': {
                'values': [True,False]
            },
            'precision': {
                'values': [32,16,'bf16']
            },
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="6DimContrSweep", entity="st7ma784")
    print(sweep_id)