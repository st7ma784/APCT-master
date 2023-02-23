import wandb

if __name__=="__main__":
    sweep_config = {
        'name':"test pruning",
        'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[5e-5]
            },
            'batch_size': {
                'values': [10]
            },
            'precision': {
                'values': ['bf16']
            },
            'embed_dim':{
                'values': [512]
            }, 
            'transformer_width':{
                'values': [512]
            },
            'logitsVersion':{
                'values':[0,1,2,3,4]
            },
            "prune":{
                'values':[True,False]
            },
            "exactlabels":{
                'values':[True,False]
            },

            "normlogits":{
                'values':[True,False]
            },
            "projection":{
                'values':["None","inv","iinv"]
            },
            'transformer_heads':{
                'values': [16]
            },
            'transformer_layers':{
                'values': [12]
            },
        }
    }
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="6DimCachespliteinSweep", entity="st7ma784")
    print(sweep_id)
