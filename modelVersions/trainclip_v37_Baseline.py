
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from functools import partial
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
# from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
import clip
from warnings import warn
import matplotlib.pyplot as plt
from CKA_test import add_colorbar 


class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                useclip_en=True,
                useclip_im=True,
                JSE=False,
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
        self.lossim=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss1=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss2=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss3=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss4=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss5=torch.nn.CrossEntropyLoss(reduction='mean')

        self.vocab_size = vocab_size
        self.automatic_optimization=False
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        print("ici")

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
        return x.contiguous()

    def orig_HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N=K.shape[0]
        return torch.add(torch.trace(K@L),torch.div(torch.sum(K)*torch.sum(L)/(N - 1) - (torch.sum(K@L) * 2 ), (N - 2)))
        
    def on_validation_epoch_start(self):
        self.eval()
        self.freeze()
    #     #import clip model here]
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self._insert_hooks()
        self.eval()
        self.model2.eval()


    def validation_step(self,batch,*args):

        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  
        self.encode_image(batch[0]) #run through main mode
        ###If your model has supervised data, then perhaps do a loss with your date here!
        self.model2.encode_image(batch[0])# to compare supervision model
        N = len(self.model1_features.values())
        M = len(self.model2_features.values())
        print("N",N)
        print("M",M)
        out=torch.stack([self.orig_HSIC(K, K) for K in self.model1_features.values()])
        self.hsic_matrix0=torch.add(self.hsic_matrix0,out) if hasattr(self, 'hsic_matrix0') else out
        out=torch.stack([self.orig_HSIC(L, L) for L in self.model2_features.values()])
        self.hsic_matrix2=torch.add(self.hsic_matrix2,out) if hasattr(self, 'hsic_matrix2') else out
        out=torch.stack([self.orig_HSIC(K, L) for K in self.model1_features.values() for L in self.model2_features.values()])
        self.hsic_matrix1=torch.add(self.hsic_matrix1,out.reshape(N,M)) if hasattr(self, 'hsic_matrix1') else out.reshape(N,M)
        self.hsic_matrix = self.hsic_matrix1 / (torch.sqrt(self.hsic_matrix0.unsqueeze(1))*torch.sqrt(self.hsic_matrix2.unsqueeze(0)))
        if not torch.isnan(self.hsic_matrix).any():
            warn("HSIC computation resulted in NANs")
            
    def on_validation_epoch_end(self,):
        self.unfreeze()
        self.train()
        self.plot_results("HSICBaseline{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="HSICBaseline{}".format(self.current_epoch), images=["HSICBaseline{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        del self.model2

    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        with torch.no_grad():
            if isinstance(out, tuple):
                out = out[0]
            #if activation shape is the same as dataloader batch size, then it is a linear layer
            if out.shape[0] == self.hparams.train_batch_size:
                print("LOGGING : ", model, name, out.shape)
                if model == "model1":
                    X = out.flatten(1)
                    self.model1_features[name] = (X @ X.t()).fill_diagonal_(0)
                elif model == "model2":
                    X = out.flatten(1)
                    self.model2_features[name] = (X @ X.t()).fill_diagonal_(0)
                else:
                    raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
       
        for name, layer in self.named_modules():
            self.handles.append(layer.register_forward_hook(partial(self._log_layer, "model1", name)))
      
        for name, layer in self.model2.named_modules():
            self.handles.append(layer.register_forward_hook(partial(self._log_layer, "model2", name)))
       
  
    def export(self):
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "CKA": self.hsic_matrix,
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix.cpu(), origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

    
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        # access your optimizers with use_pl_optimizer=False. Default is True,
        # setting use_pl_optimizer=True will maintain plugin/precision support

        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)
        logs=self.logit_scale.exp()
        #self.labels=self.labels.to(self.device)
        captions=batch[1]
        cap1,cap2,cap3,cap4,cap5=captions[0],captions[1],captions[2],captions[3],captions[4]
        cacheim=self.encode_image(batch[0])

        cacheim = cacheim / cacheim.norm(dim=1, keepdim=True)
   
        lossim= self.loss(cacheim@ caption_features1.t(),labels)
        caption_features1=self.encode_text(cap1)
        caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)
        losscap= self.loss(caption_features1@ cacheim.t(),labels)
        caption_features2=self.encode_text(cap2)
    
        caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True) 
        losscap2= self.loss(caption_features2@ cacheim.t(),labels)
        caption_features3=self.encode_text(cap3)
        # print(caption_features3.requires_grad)
        caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)
        losscap3= self.loss(caption_features3@ cacheim.t(),labels)
        caption_features4=self.encode_text(cap4)
        caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)

        losscap4= self.loss(caption_features4@ cacheim.t(),labels)
        caption_features5=self.encode_text(cap5)

        caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)
        losscap5= self.loss(caption_features5@ cacheim.t(),labels)

        self.log('train_loss', lossim+losscap+losscap2+losscap3+losscap4+losscap5, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return lossim+losscap+losscap2+losscap3+losscap4+losscap5
  
    def configure_optimizers(self):
        
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      

        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizerA), "monitor": "train_loss"}

        return [optimizerA],[lr_schedulers]
 