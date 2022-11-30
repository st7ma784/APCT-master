from PIL import Image
import torch     
import os
import pytorch_lightning as pl
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_dataset
prep=Compose([
        Resize(224, interpolation=Image.NEAREST),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
from transformers import CLIPTokenizer


# Dataset
os.environ["TOKENIZERS_PARALLELISM"]='false'

class LaionDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', T=prep, batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.T=T
        self.splits={"train":[],"val":[],"test":[]}
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        print("Loading dataset")

        def tokenize(examples):
            #print("Tokenizing: ",examples)
            try:
                return self.tokenizer(text=examples['TEXT'],
                                                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                                                    max_length = 77,           # Pad & truncate all sentences.
                                                                    padding = "max_length",
                                                                    truncation=True,
                                                                    return_attention_mask = True,   # Construct attn. masks.
                                                                    return_tensors = 'pt',     # Return pytorch tensors.
                                                                    )
                
                
            except Exception as e:
                print("Error tokenizing: ",e)
            finally:
                return examples
#            return examples        #print(load_dataset("laion/laion400m",streaming=False,split="train").map(lambda example:print(example.keys())))
        self.train=load_dataset("laion/laion400m",streaming=False,split="train").map(lambda example: tokenize(example),batched=True,num_proc=12,remove_columns=["TEXT"])
        self.train.set_transform(self.T)

        self.val=load_dataset("laion/laion400m",streaming=False,split="validation").map(lambda example: tokenize(example),batched=True,num_proc=12,remove_columns=["TEXT"])
        self.val.set_transform(self.T)
        self.test=load_dataset("laion/laion400m",streaming=False,split="test").map(lambda example: tokenize(example),batched=True,num_proc=12,remove_columns=["TEXT"])
        self.test.set_transform(self.T)
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)


    
if __name__ == '__main__':
    dm = LaionDataModule()
    print(dm.train[0])
    print(dm.val[0])
    print(dm.test[0])