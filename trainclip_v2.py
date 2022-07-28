
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm
from pytorch_lightning.callbacks import TQDMProgressBar
class myclip(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.query = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.response = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.r_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.q_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.r_ln_final = LayerNorm(transformer_width)
        self.q_ln_final = LayerNorm(transformer_width)
        self.q_text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.r_text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
    

    def encode_response(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.r_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.response(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.r_ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.r_text_projection

        return x
    def encode_query(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.q_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.query(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.q_ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.q_text_projection
        return x
    def forward(self, query, response,image_features):
        q_features = self.encode_query(query)
        r_features = self.encode_response(response)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        q_features = q_features / q_features.norm(dim=1, keepdim=True)
        r_features = r_features / r_features.norm(dim=1, keepdim=True)
        #each of these is B x D
        #in a square we get BxB Matrix of DxD matrices for logits
        #for 3 features we get BxBxB matrix of DxDxD matrices for logits
        logs=self.logit_scale.exp()
        logits_per_im=logs*torch.einsum('b...,c...,d...->bcd',image_features,q_features,r_features)

        logits_per_r= logs*torch.einsum('b...,c...,d...->bcd',r_features,image_features,q_features)

        logits_per_q= logs*torch.einsum('b...,c...,d...->bcd',q_features,r_features,image_features)
        
        return logits_per_im, logits_per_r, logits_per_q


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.q_positional_embedding, std=0.01)
        nn.init.normal_(self.r_positional_embedding, std=0.01)

        proj_std = (self.query.width ** -0.5) * ((2 * self.query.layers) ** -0.5)
        attn_std = self.query.width ** -0.5
        fc_std = (2 * self.query.width) ** -0.5
        for block in self.query.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.response.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.r_text_projection, std=self.query.width ** -0.5)
        nn.init.normal_(self.q_text_projection, std=self.query.width ** -0.5)


class LightningCLIPModule(LightningModule):
    def __init__(self,
                useclip_en=True,
                useclip_im=True,
                learning_rate: float = 2e-4,
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                **kwargs,
                ):

        super(LightningCLIPModule, self).__init__()
        self.save_hyperparameters()
        self.useclip_en = useclip_en
        self.useclip_im = useclip_im
        if self.useclip_en or self.useclip_im:
            from clip import clip
            self.stockclip,self.preprocess = clip.load("ViT-B/32", jit=False, device=self.device)
        else:
            self.preprocess=None
            self.stockclip=None
        self.clip = myclip(embed_dim= 512,
                 context_length= 77,
                 vocab_size= 50257,
                 transformer_width= 512,
                 transformer_heads= 32,
                 transformer_layers= 4)
        if self.useclip_en:
            self.encode_query=self.stockclip.encode_text
        else:
            self.encode_query=self.clip.encode_query
        self.clip.dtype=self.dtype
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.lossq = torch.nn.CrossEntropyLoss()
        self.lossr = torch.nn.CrossEntropyLoss()
        self.lossim=torch.nn.CrossEntropyLoss()

    def forward(self, query, response,im):
        return self.clip(query, response,im)
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        dims=batch[0].shape[0]
        labels=torch.diag_embed(torch.arange(dims,dtype=torch.long,device=self.device)-self.lossq.ignore_index)+self.lossq.ignore_index

        query ,response,im= batch[0],batch[1],batch[2]
        if self.useclip_im:
            im=self.stockclip.encode_image(im)
        imlogits, rlogits,qlogits = self(query, response,im)
        lossq = self.lossq(qlogits, labels)
        lossr = self.lossr(rlogits, labels)
        lossim=self.lossim(imlogits, labels)
        loss = lossq+lossr+lossim
        loss=loss/3
        loss = loss.mean()
        return {"loss": loss, "log": {"train_loss": loss}}

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.clip.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      
        return [optimizer]
import wandb
def train(config={
        "useclip_im":True,
        "useclip_en":False,
        "batchsize":64,
        "learning_rate":2e-4,
        "adam_epsilon":1e-8,
        "precision":16,
    }):
    #Load Data Module and begin training
    from BuildSpainDataSet import TriModal
    with wandb.init( project="NDIMContrSweep", entity="st7ma784", job_type="train", config=config) as run:  
        model=LightningCLIPModule(  useclip_en=config["useclip_en"],
                                    useclip_im=config["useclip_im"],
                                    learning_rate = config["learning_rate"],
                                    adam_epsilon = 1e-8)
        Dataset=TriModal(dir="MS-COCO-ES",transform=model.preprocess)
        data_module = torch.utils.data.DataLoader(Dataset,batch_size=config[100],shuffle=True,num_workers=4,pin_memory=True,drop_last=True,prefetch_factor=2)
        callbacks=[
            TQDMProgressBar()
        ]
        logtool= pytorch_lightning.loggers.WandbLogger(experiment=run)
        trainer=pytorch_lightning.Trainer(
            devices="auto",
            accelerator="auto",
            max_epochs=100,
            logger=logtool,
            callbacks=callbacks,
            gradient_clip_val=0.25,
            fast_dev_run=False,
            precision=config["precision"]
        )
        
        
        trainer.fit(model,data_module)

if __name__ == '__main__':
    config={
        "useclip_im":True,
        "useclip_en":False,
        "batchsize":64,
        "learning_rate":2e-4,
        "adam_epsilon":1e-8,
        "precision":16,
    }
    train(config)