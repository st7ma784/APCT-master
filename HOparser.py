from test_tube import HyperOptArgumentParser

class parser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="grid_search"):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        #more info at https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
        self.add_argument("--dir",default="/nobackup/projects/bdlan05/$USER/data",type=str)
        self.add_argument("--log_path",default="/nobackup/projects/bdlan05/$USER/logs/",type=str)
        self.opt_list("--learning_rate", default=0.001, type=float, options=[2e-4,1e-4,5e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--batch_size", default=8, type=float, options=[8,12,16, 20,24], tunable=True)
        self.opt_list("--JSE", default=True, type=bool, options=[True,False], tunable=True)
        self.opt_list("--precision", default=32, options=[16,'bf16'], tunable=False)
        self.opt_list("--transformer_layers", default=12, type=int, options=[3,4,5,6], tunable=True)
        self.opt_list("--transformer_heads", default=8, type=int, options=[4,8,12,16], tunable=True)
        self.opt_list("--embed_dim", default=512, type=int, options=[128,256,512,768], tunable=True)
        self.opt_list("--transformer_width", default=512, type=int, options=[128,256,512,768], tunable=True)
      
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)




if __name__== "__main__":
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams)
    for trial in hyperparams.trials(num=10):
        print(trial)
        
