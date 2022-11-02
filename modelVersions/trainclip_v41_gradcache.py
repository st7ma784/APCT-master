
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from pytorch_lightning.callbacks import TQDMProgressBar

class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                useclip_en=True,
                useclip_im=True,
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                embed_dim= 512,
                context_length= 77,
                vocab_size= 50257,
                transformer_width= 512,
                transformer_heads= 32,
                transformer_layers= 4,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        print("learning_rate",learning_rate)

        self.context_length = context_length
        self.encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
            )
        self.encode_image= VisionTransformer(
                input_resolution=224,
                patch_size=16,
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                output_dim=embed_dim
            )
        
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.loss=torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.vocab_size = vocab_size
        self.automatic_optimization=False
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        print("done")


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
  
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
 
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5

        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        #if self.useclip_im:
        image_features=self.encode_image(im)
        #image_features = image_features / image_features.norm(dim=1, keepdim=True)

        caption_features1=self.encode_text(captions1)
        #caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)

        caption_features2=self.encode_text(captions2)
        #caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True)

        caption_features3=self.encode_text(captions3)
        #caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)

        caption_features4=self.encode_text(captions4)
        #caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)

        caption_features5=self.encode_text(captions5)
        #caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)
        return image_features,caption_features1,caption_features2,caption_features3,caption_features4,caption_features5


    def calculate_loss(self, I, C1, C2, C3, C4, C5):
      
        shapes=(I.shape[0],C1.shape[0],C2.shape[0],C3.shape[0],C4.shape[0],C5.shape[0],-1)
        arr=torch.stack([I.view(I.shape[0],1,1,1,1,1,-1).expand(shapes), 
                        C1.view(1,C1.shape[0],1,1,1,1,-1).expand(shapes),
                        C2.view(1,1,C2.shape[0],1,1,1,-1).expand(shapes),
                        C3.view(1,1,1,C3.shape[0],1,1,-1).expand(shapes),
                        C4.view(1,1,1,1,C4.shape[0],1,-1).expand(shapes),
                        C5.view(1,1,1,1,1,C5.shape[0],-1).expand(shapes)], dim=-1)
        return torch.pow(torch.sub(arr,torch.mean(arr, dim=-1, keepdim=True)),2).sum(dim=-1).sum(dim=-1)
        
        
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        opt_a=self.optimizers()
        labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.loss.ignore_index))))
        labels=labels+self.loss.ignore_index
        #self.labels=self.labels.to(self.device)
        im,captions= batch[0],batch[1]
        logs=self.logit_scale.exp()

        #print(captions.shape)#Batchx 5 Capions x 
        with torch.no_grad():
            clogitsi,clogits1,clogits2,clogits3,clogits4,clogits5=self(im, captions[:,0], captions[:,1], captions[:,2], captions[:,3], captions[:,4])
        image_features=self.encode_image(im)
        #image_features = image_features / image_features.norm(dim=1, keepdim=True)
        Loss1=self.loss(self.calculate_loss(image_features, clogits1, clogits2, clogits3, clogits4, clogits5).permute(*range(-5,1))*logs,labels)
        #Loss1=self.calculate_loss(image_features, clogits1, clogits2, clogits3, clogits4, clogits5)*logs
        self.manual_backward(Loss1,retain_graph=True)
        self.log('train_loss', Loss1, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del image_features,Loss1
        caption_features1=self.encode_text(captions[:,0])
        #caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)
        Loss2=self.loss(self.calculate_loss(clogitsi, caption_features1, clogits2, clogits3, clogits4, clogits5).permute(*range(-4,2))*logs,labels)
        #Loss2=self.calculate_loss(clogitsi, caption_features1, clogits2, clogits3, clogits4, clogits5)*logs
        self.manual_backward(Loss2,retain_graph=True)
        self.log('Caption1', Loss2, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features1,Loss2
        caption_features2=self.encode_text(captions[:,1])
        #caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True)
        Loss3=self.loss(self.calculate_loss(clogitsi, clogits1, caption_features2, clogits3, clogits4, clogits5).permute(*range(-3,3))*logs,labels)
        #Loss3=self.calculate_loss(clogitsi, clogits1, caption_features2, clogits3, clogits4, clogits5)*logs
        self.manual_backward(Loss3,retain_graph=True)
        self.log('Caption2', Loss3, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features2,Loss3

        caption_features3=self.encode_text(captions[:,2])
        #caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)
        Loss4=self.loss(self.calculate_loss(clogitsi, clogits1, clogits2, caption_features3, clogits4, clogits5).permute(*range(-2,4))*logs,labels)
        #Loss4= self.calculate_loss(clogitsi, clogits1, clogits2, caption_features3, clogits4, clogits5)*logs
        self.manual_backward(Loss4,retain_graph=True)
        self.log('Caption3', Loss4, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features3,Loss4

        caption_features4=self.encode_text(captions[:,3])
        #caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)
        Loss5=self.loss(self.calculate_loss(clogitsi, clogits1, clogits2, clogits3, caption_features4, clogits5).permute(*range(-1,5))*logs,labels)
        #Loss5=self.calculate_loss(clogitsi, clogits1, clogits2, clogits3, caption_features4, clogits5)*logs
        self.manual_backward(Loss5,retain_graph=True)
        self.log('Caption4', Loss5, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features4,Loss5

        caption_features5=self.encode_text(captions[:,4])
        #caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)
        Loss6=self.loss(self.calculate_loss(clogitsi, clogits1, clogits2, clogits3, clogits4, caption_features5)*logs,labels)
        #Loss6=self.calculate_loss(clogitsi, clogits1, clogits2, clogits3, clogits4, caption_features5)*logs
        self.manual_backward(Loss6,retain_graph=True)
        self.log('Caption5', Loss6, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features5,Loss6

        opt_a.step()
        opt_a.zero_grad()
            
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      
        return [optimizer]
import wandb
def testtrainfunc(config=None,dir="/Data",devices="auto",accelerator="auto",Dataset=None):
    import time
    import os
    import socket

    from datetime import datetime as dt
    
    print('Process started {}'.format(dt.now()))
    print('NODE : {}'.format(socket.gethostname()))
    print('PID  : {}'.format(os.getpid()))
    print('Executing for 15 secs')
    time.sleep(15)
    print('Process ended {}'.format(dt.now()))
    print(config) 
    print(dir)
    print(devices)
    print(accelerator)
    print(Dataset)
    print('Process finished {}\n'.format(dt.now()))
    import torch
    print("CUDA? {}".format(torch.cuda.is_available()))
    
    print([torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())])
    with wandb.init(project="BEDETEST",entity="st7ma784",name="BEDETEST",config=config) as run:
        run.log({"test":1})  # only log first rank
def wandbtrain(config=None,dir=None,devices="auto",accelerator="auto",Dataset=None):
    if config is not None and not isinstance(config,dict):
        #print("Config is not a dict")
        config=config.__dict__
        #print("as dict: {}".format(config))
    with wandb.init(project="6DIMContrSweep",entity="st7ma784",name="6DIMContrSweep",config=config) as run:

        logtool= pytorch_lightning.loggers.WandbLogger( name="BEDEContrSweep",project="6DIMContrSweep",entity="st7ma784",experiment=run, save_dir=dir)
        #print(logtool.__dir__())
        #config=logtool.experiment.config
        #print("experiment {}".format(logtool.experiment.config))
        print("WANDB run.CONFIG {}".format(run.config))
        dir=run.config.get("dir",dir)
        train(run.config,dir,devices,accelerator,Dataset,logtool)
def train(config={
        "batch_size":16,
        "learning_rate":2e-4,
        "precision":16,
    },dir="/Data",devices="auto",accelerator="auto",Dataset=None,logtool=None):
    model=LightningCLIPModule(  learning_rate = config["learning_rate"],
                                    train_batch_size=config["batch_size"],
                                    adam_epsilon = 1e-8)
    print("Model created")
    if Dataset is None:
        from BuildSpainDataSet import COCODataModule
        #print(dir)
        #print(config)
        Dataset=COCODataModule(Cache_dir=dir,batch_size=config["batch_size"])
    Dataset.batch_size=config["batch_size"]
    print("precision {}".format(config["precision"]))

    trainer=pytorch_lightning.Trainer(
            devices=devices,
            accelerator=accelerator,
            max_epochs=100,
            #profiler="advanced",
            logger=logtool,
            strategy="ddp",
            num_nodes=1,
            #callbacks=callbacks,
            gradient_clip_val=0.25,
            precision=config.get("precision",'bf16')
    )
    if config["batch_size"] !=1:
        print("precision {}".format(config["precision"]))
        trainer.fit(model,Dataset)
    else:
        return 0 #No need to train if batch size is 1
if __name__ == '__main__':
    config={
        "batch_size":22,         #[1,4,8,16,32,64] # 13 for 8GB VRAM, 19 for 24GB VRAM
        "learning_rate":2e-4,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":'bf16',         #[32,16,'bf16']
    }
    wandbtrain(config)
