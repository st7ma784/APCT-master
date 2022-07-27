
import pytorch_lightning as pl
import os
import wandb
wandb.login()
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from clip import clip
tokenizer= AutoTokenizer.from_pretrained("gpt2")
tokenizer.vocab["</s>"] = tokenizer.vocab_size -1
tokenizer.pad_token = tokenizer.eos_token 
from torchvision import transforms
from PIL import Image
Rs=transforms.Resize((224,224),interpolation=Image.NEAREST)


class TriModal(torch.utils.data.Dataset):
    def __init__(self,dir,transform=None):
        print(tokenizer.vocab_size)
        self.dir=dir
        self.transform=transform
        self.imdir=os.path.join(dir,"images")
        self.datadir=os.path.join(dir,"data")
        filea="train_machine_spanish.xlsx"
        fileb="train_machine_english.xlsx"
        Spdataframe=pd.read_excel(os.path.join(self.datadir,filea))
        Endataframe=pd.read_excel(os.path.join(self.datadir,fileb))
        self.spanSentences=torch.cat([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in Spdataframe['caption']],dim=0)
        self.enSentences=torch.cat([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in Endataframe['caption']],dim=0)
        self.filenames=Endataframe['image_id']
    def __len__(self):
        return len(self.enSentences)
    def __getitem__(self,idx):
    
        imid=self.filenames[idx]
        imid="".join([(12-len(str(imid)))*"0"]+[str(imid)]+[".jpg"])
        img=Image.open(os.path.join(self.imdir,imid))
        img=Rs(img)
        if self.transform:
            img=self.transform(img)
        return self.enSentences[idx],self.spanSentences[idx], img

class DataSet(torch.utils.data.Dataset):
    def __init__(self,dir,filenames,transform=None):
        self.dir=dir
        self.filenames=filenames
        self.transform=transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self,idx):
        imid=self.filenames[idx]

        imid="".join([(12-len(str(imid)))*"0"]+[str(imid)]+[".jpg"])
        img=Image.open(os.path.join(self.dir,imid))
        img=Rs(img)
        if self.transform:
            img=self.transform(img)
        return img
class DataModule(pl.LightningDataModule):

    def __init__(self,dir="MS-COCO-ES",batch_size=3,):
        super().__init__(batch_size)
        self.batch_size=batch_size
        self.datadir=os.path.join(dir,"data")
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.imdir=os.path.join(dir,"images")
        self.clip,self.preprocess = clip.load("ViT-B/32", jit=False, device=self.device)

    @torch.no_grad()
    def download_data(self):
        with wandb.init(entity="st7ma784", project="NDIMTrainSweep", job_type="train-data") as run:
            data_source = wandb.Artifact("NDimtrain-data", type="dataset")
            #check datasource for the existence of the data
            print("Loading data from wandb")
            #check datasource has the data files:
            try:
                if not os.path.exists(data_source.get_path("train")):
                    self.train_dataset=torch.load(data_source.get_path("train").download(root=self.datadir))
                if not os.path.exists(data_source.get_path("validation")):
                    self.val_dataset=torch.load(data_source.get_path("validation").download(root=self.datadir))
                if not os.path.exists(data_source.get_path("test")):
                    self.test_dataset=torch.load(data_source.get_path("test").download(root=self.datadir))
            except:
                    
                filea="train_machine_spanish.xlsx"
                fileb="train_machine_english.xlsx"
                dataframe=pd.read_excel(os.path.join(self.datadir,filea),engine="openpyxl")
                sentencesa= torch.stack([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in dataframe['caption']],dim=0)
                dataframe=pd.read_excel(os.path.join(self.datadir,fileb),engine="openpyxl")

                sentencesb= torch.stack([tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in dataframe['caption']],dim=0)
                imagesids=dataframe["image_id"]
                images=[]
                dataset=DataSet(dir=self.imdir,filenames=imagesids,transform=self.preprocess)
                Dataloader=torch.utils.data.DataLoader(dataset, batch_size=1200, num_workers=4, pin_memory=True)
                for i,batch in tqdm(enumerate(Dataloader)):
                    inputs=batch.to(self.device,non_blocking=True)
                    torch.save(self.clip.encode_image(inputs), "{}.pt".format(i))
                    images.append(i)
                start=0
                datasets=[]
                for i in tqdm(images):

                    path="pretrain{}.pt".format(i)
                    data=torch.utils.data.TensorDataset( sentencesa[start:start+batch.shape[0]],sentencesb[start:start+batch.shape[0]],batch)
                    start=start+batch.shape[0]
                    torch.save(data, path)
                    datasets.append(path)
                    os.remove("{}.pt".format(i))

                
                data=torch.utils.data.ConcatDataset([torch.load(dataset) for dataset in datasets])
                #data=torch.utils.data.TensorDataset(sentencesa,sentencesb,stackedimages)

                path="pretrain.pt"
                torch.save(data, path)
                #print("datasets {} : ".format(data[0]))
                train_size = int(0.8 * len(data))
                val_size = int(0.1 *  len(data))
                test_size= len(data) - (train_size+val_size)

                self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(data, [train_size, val_size,test_size])
                torch.save(self.train_dataset, "train.pt")
                torch.save(self.val_dataset, "validation.pt")
                torch.save(self.test_dataset, "test.pt")
                data_source.add_file("train.pt", name="train")
                data_source.add_file("validation.pt", name="validation")
                data_source.add_file("test.pt", name="test")
                run.log_artifact(data_source)
    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            #check if "espretrain.pt") exists in the directory
            if os.path.exists("train.pt"):
                self.train_dataset=torch.load("train.pt")
            else:
                self.download_data()
            
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            #check if esprevalidation.pt exists in the directory
            if os.path.exists("validation.pt"):
                self.val_dataset=torch.load("validation.pt")
            else:
                self.download_data()
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            #check for espretest.pt in the directory
            if os.path.exists("test.pt"):
                self.test_dataset=torch.load("test.pt")
            else:

                self.download_data()

        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True)
if __name__=="__main__":
    data=DataModule(batch_size=3)
    data.download_data()
