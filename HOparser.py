from test_tube import HyperOptArgumentParser

class parser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        #more info at https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
        self.add_argument("--dir",default="/nobackup/projects/bdlan05/smander3/data",type=str)
        self.add_argument("--log_path",default="/nobackup/projects/bdlan05/smander3/logs/",type=str)
        self.opt_list("--learning_rate", default=0.00001, type=float, options=[1e-3,1e-5, 1e-4,], tunable=True)
        self.opt_list("--batch_size", default=10, type=int, options=[6,10,14], tunable=True)
        self.opt_list("--JSE", default=0, type=int, options=[0], tunable=True)
        self.opt_list("--precision", default=16, options=[16], tunable=False)
        self.opt_list("--codeversion", default=5, type=int, options=[5], tunable=False)
        self.opt_list("--transformer_layers", default=5, type=int, options=[3,4,5,6], tunable=True)
        self.opt_list("--transformer_heads", default=16, type=int, options=[16], tunable=True)
        self.opt_list("--embed_dim", default=512, type=int, options=[128,512], tunable=True)
        self.opt_list("--transformer_width", default=512, type=int, options=[128,512], tunable=True)
        self.opt_list("--devices", default=1, type=int, options=[1], tunable=False)
        self.opt_list("--accelerator", default='gpu', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)



# Testing to check param outputs
if __name__== "__main__":
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    for trial in hyperparams.generate_trials(10):
        print(trial)
        
