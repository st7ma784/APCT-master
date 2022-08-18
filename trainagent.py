


   
   
import wandb
from trainclip_v2 import train
from BuildSpainDataSet import COCODataModule
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import argparse
import wandb
from functools import partial
wandb.login()
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", default="futbumne",nargs="?", type=str)
parser.add_argument("--data_dir", default="/Data",nargs="?", type=str)
parser.add_argument("--devices", default="auto",nargs="?", type=str)
parser.add_argument("--accelerator", default="auto",nargs="?", type=str)

p = parser.parse_args()
preprocess=Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
Dataset=COCODataModule(Cache_dir=p.data_dir,T=preprocess)
Dataset.prepare_data()
assert len(Dataset)>0,"Dataset is empty"
train=partial(train,dir=p.data_dir,devices=p.devices, accelerator=p.accelerator,Dataset=Dataset)
wandb.agent(sweep_id=p.sweep_id, project="6DimContrSweep", entity="st7ma784",function=train)
