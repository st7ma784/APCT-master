from torchvision import transforms
from PIL import Image
import torch     
import os
import zipfile
from pySmartDL import SmartDL
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from torchvision.datasets import CocoCaptions
T= transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),transforms.ToTensor()])
from transformers import AutoTokenizer
import time
from pathlib import Path
class COCODataset(CocoCaptions):
    def __init__(self, root, annFile, tokenizer, *args, **kwargs):
        print('Loading COCO dataset')
        self.tokenizer=tokenizer
        #check if root and annfile exist
        assert(os.path.exists(root),'root does not exist')
        #print('Error: root directory does not exist: {}'.format(root))
        assert(os.path.exists(annFile),'annFile does not exist')
        #print('Error: annFile does not exist: {}'.format(annFile))
        super().__init__(root, annFile, *args, **kwargs)
        print('Done')
        #print(self.ids)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx: int):
        try:
            img, target= super().__getitem__(idx)
        except Exception as e:
            print(e)
            print('Error loading image:', index)
            return None
        target=torch.cat([self.tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in target[:5]],dim=0)
        return img,target





# Dataset

class COCODataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='./', T=None, batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.T=T
        self.splits={"train":[],"val":[],"test":[]}
        self.tokenizer=AutoTokenizer.from_pretrained("gpt2",cache_dir=self.data_dir)
        self.tokenizer.vocab["</s>"] = self.tokenizer.vocab_size -1
        self.tokenizer.pad_token = self.tokenizer.eos_token 
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir,exist_ok=True)
        urls=['http://images.cocodataset.org/zips/train2014.zip',
                'http://images.cocodataset.org/zips/val2014.zip',
                'http://images.cocodataset.org/zips/test2015.zip',
                'http://images.cocodataset.org/zips/train2017.zip',
                'http://images.cocodataset.org/zips/val2017.zip',
                'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                ]

        objs=[]
        for url in urls:
            print("url:",url)
            name=str(url).split('/')[-1]
        
            
            location=self.data_dir # if name.startswith("annotations") else self.ann_dir
            #print("Location", location) #/Data/train2014.zip
            #time.sleep(5)
            #print('Downloading',url)
            if name.endswith(".zip"):
                name=name[:-4]
            if name.startswith("train"):
                self.splits['train'].append(name)
            elif name.startswith("val"):
                self.splits['val'].append(name)
            elif name.startswith("test"):
                self.splits['test'].append(name)
            obj=SmartDL(url,os.path.join(location,str(url).split('/')[-1]),progress_bar=False)
            obj.FileName=name
            if not os.path.exists(obj.get_dest()):

                objs.append(obj)#SmartDL(url, self.data_dir,)
                obj.start(blocking=False)
                print("obj Path ",obj.get_dest())
        for obj in objs:
            while not obj.isFinished():
                #print("Speed: %s" % obj.get_speed(human=True))
                print("Eta: %s" % obj.get_eta(human=True))
                time.sleep(5)
            if obj.isSuccessful():
                print("Downloaded: %s" % obj.get_dest())

            path = obj.get_dest()
            if obj.FileName.startswith("annotations"):
                print("Extracting annotations")
            else:
                print("Extracting images")
            print("path:",path)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                try:
                    zip_ref.extractall(self.data_dir)
                except:
                    print("Error extracting annotations")
                    print("path:",path)
                    print("ann_dir:",self.ann_dir)
                #wget.download("http://images.cocodataset.org/zips/train2014.zip",out=self.cocodir)
                #walk over output to avoid HEC cleanup
                for root, dirs, files in os.walk(zip_ref.namelist()[0]):
                    for file in files:
                        Path(os.path.join(root, file)).touch()
 
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        print("Entered COCO datasetup")
        
        if stage == 'fit' or stage is None:
            TrainSets=[]
            for version in self.splits['train']:
                
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                dir=os.path.join(self.data_dir,version)

                #time.sleep(2)
                dset=COCODataset(root=dir, annFile=annfile, tokenizer=self.tokenizer, transform=self.T)
                print("dset:",dset.__dir__())

                if len(dset)>0:
                    TrainSets.append(dset)
            assert len(TrainSets)>0,"No train sets found"
            self.train = ConcatDataset(TrainSets)

            ValSets=[]
            for version in self.splits['val']:
                print("BUILDING SPLIT : ",version)
                
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                dir=os.path.join(self.data_dir,version)
                print("annfile:",annfile)
                print("dir:",dir)
                ValSets.append(COCODataset(root=dir, annFile=annfile, tokenizer=self.tokenizer, transform=self.T))
            self.val = ConcatDataset(ValSets)
            # torch.save(self.train,"train.pt")
            # torch.save(self.val,"val.pt")    
        if stage == 'test':
            TestSets=[]
            for version in self.splits['test']:
                print("BUILDING SPLIT : ",version)
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                dir=os.path.join(self.data_dir,version)
                
                print("annfile:",annfile)
                print("dir:",dir)
                TestSets.append(COCODataset(root=dir, annFile=annfile,tokenizer=self.tokenizer, transform=self.T))
            self.test = ConcatDataset(TestSets)


    
