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
                'values':[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
            },
            'batch_size': {
                'values': [2,4,8,10,12,24,48]
            },
            'precision': {
                'values': [32,16]
            },
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="6DimContrSweep", entity="st7ma784")
    print(sweep_id)