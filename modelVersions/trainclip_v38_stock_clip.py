
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from functools import partial
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
# from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
import clip
from warnings import warn
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

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
        self.JSE=JSE
        self.gelu=nn.GELU()
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
        self.model1,_ = clip.load("ViT-B/32", device=self.device)
        self.model1.train()

        self.vocab_size = vocab_size
        self.automatic_optimization=False
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        self.handles=[]


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
 
    def batch_HSIC2(self,K):
        a=torch.sum(K,dim=-1)
        b=torch.sum(K,dim=-2)
        c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[1] - 1),torch.sum(a*b,dim=1),alpha=2)
        output=torch.add(torch.einsum('a...->a',torch.pow(K,2)),torch.div(c,(K.shape[1] - 2)))
        return output
                
    def batch_HSIC3(self,K,L):
        a=torch.sum(L,dim=-1)
        b=torch.sum(K,dim=-2)
        c=torch.sub(torch.sum(b,dim=-1)*torch.sum(a,dim=-1)/(K.shape[1] - 1),torch.sum(b*a,dim=1),alpha=2)
        return torch.add(torch.einsum('abc->a',K*L),torch.div(c,(K.shape[1] - 2)))

    def on_validation_epoch_start(self):
        self.eval()
        self.naninfcount=0
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self.model2.eval()
        self._insert_hooks()
        self.eval()
        
    def validation_step(self,batch,*args):

        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        self.encode_image(batch[0]) #run through main mode
        self.encode_text(batch[1][:,0])

        self.model2.encode_image(batch[0])# to compare supervision model
        self.model2.encode_text(batch[1][:,0])
        a=torch.stack(list(self.model1_features.values()))
        if not hasattr(self,'hsic_matrix0'):
            self.hsic_matrix0=torch.zeros((a.shape[0]),device=self.device)
        self.hsic_matrix0=torch.add(self.hsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.stack(list(self.model2_features.values()))
        if not hasattr(self,'hsic_matrix2'):
            self.hsic_matrix2=torch.zeros((a.shape[0]),device=self.device)
        self.hsic_matrix2=torch.add(self.hsic_matrix2,self.batch_HSIC2(a))
        joint_HSIC=torch.stack(list(map(lambda X: self.batch_HSIC3(a,X),list(self.model1_features.values()))))
        if not hasattr(self,'hsic_matrix1'):
            self.hsic_matrix1=torch.zeros(joint_HSIC.shape,device=self.device)
        self.hsic_matrix1=torch.add(self.hsic_matrix1,joint_HSIC) 
    def on_validation_epoch_end(self,):
        self.unfreeze()
        self.train()
        self.plot_results("HSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="HSIC{}".format(self.current_epoch), images=["HSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        print(self.naninfcount)
        del self.model2
        del self.hsic_matrix0
        del self.hsic_matrix1
        del self.hsic_matrix2
        

    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        if isinstance(out, tuple):
            out=out[0]       
            # print("permuted")
        if out.shape[0] == self.hparams.train_batch_size:
            self.__store(out,name,model,layer)
        
        elif out.shape[1] == self.hparams.train_batch_size:
            self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)

    def __store(self,out,name, model,layer):
        X = out.flatten(1)
        X= (X @ X.t()).fill_diagonal_(0)
        if (torch.isnan(X).any() or torch.isinf(X).any()):
            self.naninfcount+=1
            if self.current_epoch==0 and hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, std=0.02)
        if model == "model1":
            #if name already exists in dictionary, change name to name+1
            while name in self.model1_features:
                name=name+"1"
            self.model1_features[name] = X

        elif model == "model2":
            while name in self.model1_features:
                name=name+"1"
            self.model2_features[name] = X

        else:
            raise RuntimeError("Unknown model name for _log_layer.")
    def _insert_hooks(self):
        self.handles=[]
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules()])
        a=len(self.handles)
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        b=len(self.handles)-a
        return a,b
  
    def export(self):
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "CKA":self.hsic_matrix1 / (torch.sqrt(self.hsic_matrix0.unsqueeze(1))*torch.sqrt(self.hsic_matrix2.unsqueeze(0))),
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        t=self.hsic_matrix0.unsqueeze(1)*self.hsic_matrix2.unsqueeze(0)
        print(torch.sum(torch.abs(t)==t))
        r=torch.sqrt(torch.abs(t))
        r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
        hsic_matrix = self.hsic_matrix1 / r
        if not torch.isnan(hsic_matrix).any():
            warn("HSIC computation resulted in NANs")
        im = ax.imshow(hsic_matrix.cpu(), origin='lower', cmap='magma')
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

    
    def training_step(self, batch, batch_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True,
        # setting use_pl_optimizer=True will maintain plugin/precision support
        opt_a,opt_b = self.optimizers()

        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)
        
        logitsa=self.model1.encode_image(batch[0])
        logitsb=self.model1.encode_text(batch[1][np.random.randint(0,5)])
        logitsa=logitsa/logitsa.norm(dim=1,keepdim=True)
        logitsb=logitsb/logitsb.norm(dim=1,keepdim=True)
        lossa=self.loss(logitsa,labels)
        lossb=self.loss(logitsb,labels)
        loss=lossa+lossb /2
        self.log('stock_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.manual_backward(loss)

        opt_b.step()
        opt_b.zero_grad()

        labels=torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.loss.ignore_index)
        logs=self.logit_scale.exp()
        for i in range(3):
            labels=torch.diag_embed(labels)
        labels=labels+self.loss.ignore_index
        #self.labels=self.labels.to(self.device)
        with torch.no_grad():
            cache=self.encode_text(batch[1].flatten(start_dim=0,end_dim=1)).unflatten(0,(batch[1].shape[0],5),)
            cache=cache/cache.norm(dim=-1, keepdim=True)
            cache1=cache[:,0]
            cache2=cache[:,1]#.to(torch.device("cpu"),non_blocking=True)
            cache3=cache[:,2]#.to(torch.device("cpu"),non_blocking=True)
            cache4=cache[:,3]#.to(torch.device("cpu"),non_blocking=True)
            cache5=cache[:,4]#.to(torch.device("cpu"),non_blocking=True)
            del cache
        cacheim=self.encode_image(batch[0])
        # if self.JSE:
        #     JSEFactor=1-(4/torch.sum(torch.stack([cacheim,cache1,cache2,cache3,cache4,cache5],dim=0).pow(2),dim=0))
        #     print(JSEFactor)
        #     #cacheim=torch.mul(cacheim,JSEFactor)
        #     #cacheim=self.gelu(cacheim)
        #     del JSEFactor

        cacheim = cacheim / cacheim.norm(dim=1, keepdim=True)
        loss = self.loss(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)
        self.manual_backward(loss,retain_graph=True)
        cacheim=cacheim.detach()#.to(torch.device("cpu"),non_blocking=True)
        self.log('imloss', loss, prog_bar=True,enable_graph=False,rank_zero_only=True)


        del loss,batch[0]        

        cap1,cap2,cap3,cap4,cap5=batch[0][:,0],batch[0][:,1],batch[0][:,2],batch[0][:,3],batch[0][:,4]
        del batch

        caption_features1=self.encode_text(cap1)
        #print(caption_features1.requires_grad)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([caption_features1,cache2,cache3,cache4,cache5,cacheim]),2),dim=0))
            #print(JSEFactor)

            caption_features1=torch.mul(caption_features1,.96)
            caption_features1=self.gelu(caption_features1)
            #del JSEFactor
        caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)
        loss = self.loss(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,caption_features1,cache2)),labels)

        self.manual_backward(loss,retain_graph=True)
        self.log('caption1', loss, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features1,loss,cap1


        caption_features2=self.encode_text(cap2)
        #print(caption_features2.requires_grad)
        if self.JSE:

            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([cache1,caption_features2,cache3,cache4,cache5,cacheim]),2),dim=0))
            # print(JSEFactor)

            caption_features2=torch.mul(caption_features2,JSEFactor)
            caption_features2=self.gelu(caption_features2)
            del JSEFactor
        caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True) 
        loss = self.loss(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,caption_features2)),labels)        
        self.manual_backward(loss,retain_graph=True)
        self.log('caption2', loss, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features2,loss,cap2


        caption_features3=self.encode_text(cap3)
        # print(caption_features3.requires_grad)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([cache1,cache2,caption_features3,cache4,cache5,cacheim]),2),dim=0))
            caption_features3=torch.mul(caption_features3,JSEFactor)
            caption_features3=self.gelu(caption_features3)
            del JSEFactor
        caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)
        loss = self.loss(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",caption_features3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)        
        self.manual_backward(loss,retain_graph=True)
        self.log('caption3', loss,  prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features3,loss,cap3 


        caption_features4=self.encode_text(cap4)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([cache1,cache2,cache3,caption_features4,cache5,cacheim]),2),dim=0))
            caption_features4=torch.mul(caption_features4,JSEFactor)
            caption_features4=self.gelu(caption_features4)
            del JSEFactor

        caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)
        #print(caption_features4.requires_grad)
        loss = self.loss(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,caption_features4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)        
        self.manual_backward(loss,retain_graph=True)
        self.log('caption4', loss,  prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features4,loss,cap4


        caption_features5=self.encode_text(cap5)
        if self.JSE:
            JSEFactor=-torch.div(4,torch.sum(torch.pow(torch.stack([cache1,cache2,cache3,cache4,caption_features5,cacheim]),2),dim=0))
            caption_features5=torch.mul(caption_features5,torch.add(JSEFactor,1))
            caption_features5=self.gelu(caption_features5)
            del JSEFactor
        caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)
        #print(caption_features5.requires_grad)
        loss = self.loss(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,caption_features5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)        
        self.manual_backward(loss,retain_graph=True)
        self.log('caption5', loss, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features5,loss,cap5


        opt_a.step()
        opt_a.zero_grad()
        #        self.backward(0)
            
    def configure_optimizers(self):
        
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizerB=torch.optim.Adam(self.model1.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        lr_schedulerA = {"scheduler": ReduceLROnPlateau(optimizerA), "monitor": "train_loss"}
        lr_schedulerB = {"scheduler": ReduceLROnPlateau(optimizerB), "monitor": "train_loss"}
        
        return [optimizerA,optimizerB],[lr_schedulerA,lr_schedulerB]
