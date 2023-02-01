from torchvision import transforms
from PIL import Image
import torch     
import os
import zipfile
from pySmartDL import SmartDL
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
prep=Compose([
        Resize(224, interpolation=Image.NEAREST),
        CenterCrop(224),
        #Note: the standard  lambda function here is not supported by pytorch lightning
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
T= transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),transforms.ToTensor()])
from transformers import AutoTokenizer,GPT2Tokenizer, CLIPTokenizer
import time
from pathlib import Path
class COCODataset(CocoCaptions):
    def __init__(self, root, annFile, tokenizer, instances=None,*args, **kwargs):
        #print('Loading COCO dataset')
        self.tokenizer=tokenizer
        if os.getenv('ISHEC',False):
            for root, dirs, files in os.walk(root):
                for file in files:
                    Path(os.path.join(root, file)).touch()
            Path(annFile).touch()

        if not os.path.exists(root):
            print("root does not exist {}".format(root))
        #print('Error: root directory does not exist: {}'.format(root))
        if not os.path.exists(annFile):
            print('annFile does not exist {}'.format(annFile)) 
        #print('Error: annFile does not exist: {}'.format(annFile))
        super().__init__(root, annFile, *args, **kwargs)
        #print('Done')
        if instances is not None:
            from pycocotools.coco import COCO
            self.instances=COCO(instances)
        #print(self.ids)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx: int):
        try:
            img, target= super().__getitem__(idx)
        except Exception as e:
            print(e)
            print('Error loading image:', idx)
            return None
        id=self.ids[idx]
        ids=self.instances.getAnnIds(imgIds=id)

        instance= self.instances.loadAnns(ids)

        #print(id)
        #print(ids)
        #print("instances",instance[0].get("category_id",-100))
        try:
            i=instance[0].get("category_id",-100)
        except:
            i=-100

        target=torch.cat([self.tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'] for sent in target[:5]],dim=0)
        return img,target,i





# Dataset
os.environ["TOKENIZERS_PARALLELISM"]='true'

class COCODataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', T=prep, batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.T=T
        self.splits={"train":[],"val":[],"test":[]}
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        # try: 
        #     self.tokenizer=AutoTokenizer.from_pretrained("gpt2",cache_dir=self.data_dir)
        # except ValueError as e:
        #     from transformers import GPT2Tokenizer
        #     tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=self.data_dir)
        #     tok.save_pretrained(self.data_dir)
        # finally:
        #     self.tokenizer=AutoTokenizer.from_pretrained("gpt2",cache_dir=self.data_dir)
        #self.tokenizer.vocab["</s>"] = self.tokenizer.vocab_size -1
        #self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.prepare_data()
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
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir,exist_ok=True)
        urls=['https://images.cocodataset.org/zips/train2014.zip',
                'https://images.cocodataset.org/zips/val2014.zip',
                'https://images.cocodataset.org/zips/test2015.zip',
                'https://images.cocodataset.org/zips/train2017.zip',
                'https://images.cocodataset.org/zips/val2017.zip',
                'https://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                ]

        objs=[]
        for url in urls:
            #print("url:",url)
            name=str(url).split('/')[-1]
            location=self.data_dir # if name.startswith("annotations") else self.ann_dir
            #print("Location", location) #/Data/train2014.zip
            #time.sleep(5)
            #print('Downloading',url)
            obj=SmartDL(url,os.path.join(location,name),progress_bar=False, verify=False)
            obj.FileName=name
            if name.endswith(".zip"):
                name=name[:-4]
            if name.startswith("train"):
                self.splits['train'].append(name)
            elif name.startswith("val"):
                self.splits['val'].append(name)
            elif name.startswith("test"):
                self.splits['test'].append(name)
            if not os.path.exists(os.path.join(location,name)) and not (name.startswith("annotations") and os.path.exists(os.path.join(location,"annotations"))):
                print(os.path.join(location,name))
                objs.append(obj)
                obj.start(blocking=False,  )#There are security problems with Hostename 'images.cocodataset.org' and Certificate 'images.cocodataset.org' so we need to disable the SSL verification
        for obj in objs:
            while not obj.isFinished():
                time.sleep(5)
            if obj.isSuccessful():
                print("Downloaded: %s" % obj.get_dest())
            path = obj.get_dest()
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    try:
                        zip_ref.extractall(self.data_dir)
                    except Exception as e:
                        print(e)
                        print("Error extracting zip" ,path)
                        continue        
                    for root, dirs, files in os.walk(zip_ref.namelist()[0]):
                        for file in files:
                            Path(os.path.join(root, file)).touch()
 
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        
        if stage == 'fit' or stage is None:
            TrainSets=[]
            for version in self.splits['train']:
                
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                instancesfile=os.path.join(self.ann_dir,'{}_{}.json'.format('instances',version))
                dir=os.path.join(self.data_dir,version)
                if not os.path.exists(annfile):
                    print("Missing annotation file",annfile)
                
                print("Loading train dataset",annfile)
                #time.sleep(2)
                dset=COCODataset(root=dir, annFile=annfile, tokenizer=self.tokenizer,instances=instancesfile, transform=self.T)
                
                #print("dset:",dset.__dir__())

                if len(dset)>0:
                    TrainSets.append(dset)
            assert len(TrainSets)>0,"No train sets found"
            self.train = ConcatDataset(TrainSets)

            ValSets=[]
            for version in self.splits['val']:
                #print("BUILDING SPLIT : ",version)
                
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                instancesfile=os.path.join(self.ann_dir,'{}_{}.json'.format('instances',version))
                dir=os.path.join(self.data_dir,version)
                #print("annfile:",annfile)
                #print("dir:",dir)
                ValSets.append(COCODataset(root=dir, annFile=annfile, tokenizer=self.tokenizer,instances=instancesfile, transform=self.T))
            self.val = ConcatDataset(ValSets)
            # torch.save(self.train,"train.pt")
            # torch.save(self.val,"val.pt")    
        if stage == 'test':
            TestSets=[]
            for version in self.splits['test']:
                #print("BUILDING SPLIT : ",version)
                annfile=os.path.join(self.ann_dir,'{}_{}.json'.format('captions',version))
                instancesfile=os.path.join(self.ann_dir,'{}_{}.json'.format('instances',version))
                dir=os.path.join(self.data_dir,version)
                
                #print("annfile:",annfile)
                #print("dir:",dir)
                TestSets.append(COCODataset(root=dir, annFile=annfile,tokenizer=self.tokenizer,instances=instancesfile, transform=self.T))
            self.test = ConcatDataset(TestSets)


    
