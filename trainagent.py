
import wandb
from trainclip_v2 import wandbtrain
from BuildSpainDataSet import COCODataModule
import argparse
from functools import partial
wandb.login()
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", default="futbumne",nargs="?", type=str)
parser.add_argument("--data_dir", default="/Data",nargs="?", type=str)
parser.add_argument("--devices", default="auto",nargs="?", type=str)
parser.add_argument("--accelerator", default="auto",nargs="?", type=str)

p = parser.parse_args()

Dataset=COCODataModule(Cache_dir=p.data_dir)
Dataset.prepare_data()
Dataset.setup()
train=partial(wandbtrain,dir=p.data_dir,devices=p.devices, accelerator=p.accelerator,Dataset=Dataset)
wandb.agent(sweep_id=p.sweep_id, project="6DimCachespliteinSweep", entity="st7ma784",function=train)
