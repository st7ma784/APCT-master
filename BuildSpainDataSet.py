
import pytorch_lightning as pl
import os
import wandb
wandb.login()
import pandas as pd
import torch
from transformers import AutoTokenizer

tokenizer= AutoTokenizer.from_pretrained("t5-base")
tokenizer.vocab["</s>"] = 32099
class DataModule(pl.LightningDataModule):

    def __init__(self,dir="MSCOCOES",batch_size=3,):
        super().__init__(batch_size)
        self.batch_size=batch_size
        self.datadir=os.path.join(dir,"data")
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    @torch.no_grad()
    def download_data(self):
        with wandb.init(entity="st7ma784", project="T5TrainSweep", job_type="spain-pretrain-data") as run:
            data_source = wandb.Artifact("spanish-pretrain-data", type="dataset")
            #check datasource for the existence of the data
            print("Loading data from wandb")
            #check datasource has the data files:
            try:
                if not os.path.exists(data_source.get_path("pretrain-spain")):
                    self.train_dataset=torch.load(data_source.get_path("pretrain-spain").download(root=self.datadir))
                if not os.path.exists(data_source.get_path("prevalidation-spain")):
                    self.val_dataset=torch.load(data_source.get_path("prevalidation-spain").download(root=self.datadir))
                if not os.path.exists(data_source.get_path("pretest-spain")):
                    self.test_dataset=torch.load(data_source.get_path("pretest-spain").download(root=self.datadir))
            except:
                    
                filea="train_machine_spanish.xlsx"
                fileb="train_machine_english.xlsx"
                dataframe=pd.read_excel(os.path.join(self.datadir,filea))
                sentencesa= torch.stack([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in dataframe['caption']],dim=0)
                dataframe=pd.read_excel(os.path.join(self.datadir,fileb))

                sentencesb= torch.stack([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in dataframe['caption']],dim=0)
                data=torch.utils.data.TensorDataset(sentencesa,sentencesb)
                path="pretrain.pt"
                torch.save(data, path)
                #print("datasets {} : ".format(data[0]))
                train_size = int(0.8 * len(data))
                val_size = int(0.1 *  len(data))
                test_size= len(data) - (train_size+val_size)

                self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(data, [train_size, val_size,test_size])
                torch.save(self.train_dataset, "espretrain.pt")
                torch.save(self.val_dataset, "esprevalidation.pt")
                torch.save(self.test_dataset, "espretest.pt")
                data_source.add_file("espretrain.pt", name="pretrain-spain")
                data_source.add_file("esprevalidation.pt", name="prevalidation-spain")
                data_source.add_file("espretest.pt", name="pretest-spain")
                run.log_artifact(data_source)
    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            #check if "espretrain.pt") exists in the directory
            if os.path.exists("espretrain.pt"):
                self.train_dataset=torch.load("espretrain.pt")
            else:
                self.download_data()
            
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            #check if esprevalidation.pt exists in the directory
            if os.path.exists("esprevalidation.pt"):
                self.val_dataset=torch.load("esprevalidation.pt")
            else:
                self.download_data()
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            #check for espretest.pt in the directory
            if os.path.exists("espretest.pt"):
                self.test_dataset=torch.load("espretest.pt")
            else:

                self.download_data()

        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
if __name__=="__main__":
    data=DataModule(batch_size=3)
    data.download_data()