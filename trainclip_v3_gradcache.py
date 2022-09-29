
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from pytorch_lightning.callbacks import TQDMProgressBar
from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam

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
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
       


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

    def training_step(self, batch, batch_idx,optimizer_idx=0):
        # access your optimizers with use_pl_optimizer=False. Default is True,
        # setting use_pl_optimizer=True will maintain plugin/precision support
        opt_a = self.optimizers()

        labels=torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.loss.ignore_index)
        logs=self.logit_scale.exp()
        for i in range(3):
            labels=torch.diag_embed(labels)
        labels=labels+self.loss.ignore_index
        #self.labels=self.labels.to(self.device)
        im,captions= batch[0],batch[1]
        cap1,cap2,cap3,cap4,cap5=captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4]
        with torch.no_grad():
            cache=self.encode_text(captions.flatten(start_dim=0,end_dim=1)).unflatten(0,(captions.shape[0],5),)
            
            cache=cache/cache.norm(dim=-1, keepdim=True)
            cache1=cache[:,0]
            cache2=cache[:,1]#.to(torch.device("cpu"),non_blocking=True)
            cache3=cache[:,2]#.to(torch.device("cpu"),non_blocking=True)
            cache4=cache[:,3]#.to(torch.device("cpu"),non_blocking=True)
            cache5=cache[:,4]#.to(torch.device("cpu"),non_blocking=True)
            del cache,captions
        
        cacheim=self.encode_image(im)
        cacheim = cacheim / cacheim.norm(dim=1, keepdim=True)
        lossim = self.loss(logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',cacheim,cache1,cache2,cache3,cache4,cache5),labels)
        self.log('imloss', lossim, prog_bar=True,enable_graph=False,rank_zero_only=True)
        self.manual_backward(lossim,retain_graph=True)

        cacheim=cacheim.detach()#.to(torch.device("cpu"),non_blocking=True)

        del im,lossim

        caption_features1=self.encode_text(cap1)
        caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)
        loss1 = self.loss(logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features1,cache2,cache3,cache4,cache5,cacheim),labels)
        self.log('caption1', loss1, prog_bar=True,enable_graph=False,rank_zero_only=True)
        #self.all_gather(loss1,sync_grads=True)

        self.manual_backward(loss1,retain_graph=True)

        del caption_features1,loss1

        caption_features2=self.encode_text(cap2)
        caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True)
        loss2 = self.loss(logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features2,cache3,cache4,cache5,cacheim,cache1)
,labels)
        #self.all_gather(loss2,sync_grads=True)

        self.log('caption2', loss2, prog_bar=True,enable_graph=False,rank_zero_only=True)
        self.manual_backward(loss2,retain_graph=True)


        del caption_features2,loss2
        caption_features3=self.encode_text(cap3)
        caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)
        loss3 = self.loss(logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features3,cache4,cache5,cacheim,cache1,cache2)
,labels)
        #self.all_gather(loss3,sync_grads=True)

        self.log('caption3', loss3, prog_bar=True,rank_zero_only=True,enable_graph=False)
        self.manual_backward(loss3,retain_graph=True)

        del caption_features3,loss3

        caption_features4=self.encode_text(cap4)
        caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)
        loss4 = self.loss(logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features4,cache5,cacheim,cache1,cache2,cache3)
,labels)
        #self.all_gather(loss4,sync_grads=True)

        self.log('caption4', loss4, prog_bar=True,enable_graph=False,rank_zero_only=True)
        self.manual_backward(loss4,retain_graph=True)

        del caption_features4,loss4

        caption_features5=self.encode_text(cap5)
        caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)
        loss5 = self.loss(logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features5,cache1,cache2,cache3,cache4,cacheim), labels)
        #self.all_gather(loss5,sync_grads=True)

        self.manual_backward(loss5,retain_graph=True)

        self.log('caption5', loss5, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features5,loss5
        opt_a.step()
        opt_a.zero_grad()
        #        self.backward(0)
            
    def configure_optimizers(self):
        
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      

        return [optimizerA]
import wandb
def wandbtrain(config=None,dir="/Data",devices="auto",accelerator="auto",Dataset=None):
    with wandb.init(project="6DIMCacheSweep",entity="st7ma784",config=config) as run:

        logtool= pytorch_lightning.loggers.WandbLogger( project="6DIMCacheSweep",entity="st7ma784",experiment=run, save_dir=dir)
        #print(logtool.__dir__())
        config=logtool.experiment.config
        print("WANDB CONFIG",config)
        train(config,dir,devices,accelerator,Dataset,logtool)
def train(config={
        "batch_size":16,
        "learning_rate":2e-3,
        "precision":16,
    },dir="/Data",devices="auto",accelerator="auto",Dataset=None,logtool=None):
    model=LightningCLIPModule(  learning_rate = config["learning_rate"],
                                    train_batch_size=config["batch_size"],
                                    adam_epsilon = 1e-8)
    if Dataset is None:
        from BuildSpainDataSet import COCODataModule

        Dataset=COCODataModule(Cache_dir=dir,batch_size=config["batch_size"])
    Dataset.batch_size=config["batch_size"]
    trainer=pytorch_lightning.Trainer(
            devices=devices,
            accelerator=accelerator,
            max_epochs=100,
            #profiler="advanced",
            logger=logtool,
            strategy="ddp",#deepspeed_stage_1
            #callbacks=callbacks,
            #accumulate_grad_batches=8,
            #gradient_clip_val=0.25,
            precision=config["precision"]
    )
    if config["batch_size"] !=1:
        
        trainer.fit(model,Dataset)
    else:
        return 0 #No need to train if batch size is 1
if __name__ == '__main__':
    config={
        "batch_size":23,         #[1,4,8,16,32,64] #V2: 13 for 8GB VRAM, 22 for 24GB VRAM (ETA 00:48:00)
        #                                          #v3: 19 for 10GB VRAM (ETA 1:46:00),   23 for 24GB VRAM  
        # in 2 dim, 19 : 23 Batchs is the difference of 168 Samples, in 6 dim its 144 Million. 
        "learning_rate":1e-3,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":'bf16',         #[32,16,'bf16']
    }
    wandbtrain(config)
