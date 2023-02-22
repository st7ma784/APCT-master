
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from functools import partial
import clip
from warnings import warn
import matplotlib.pyplot as plt
from CKA_test import add_colorbar 
from sklearn.linear_model import LogisticRegression

class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                logitsversion=0,
                normlogits=True,
                proj='inv',
                prune=True,
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
        self.loss=torch.nn.CrossEntropyLoss(reduction='sum')

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.handles=[]
        self.features=[]
        
        self.labels=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        print("done")
        from model.LossCalculation import calculate_lossStock as sl
        self.calculate_lossStock=sl
        if logitsversion==0:
            from model.LossCalculation import calculate_loss as cl
        elif logitsversion==1: 
            from model.LossCalculation import calculate_loss2 as cl
        elif logitsversion==2: 
            from model.LossCalculation import calculate_loss3 as cl
        elif logitsversion==3:
            from model.LossCalculation import calculate_loss4 as cl
        elif logitsversion==4:
            from model.LossCalculation import calculate_loss5 as cl
        else:
            from model.LossCalculation import calculate_loss as cl
        self.calculate_loss=cl
        self.norm=normlogits
        self.proj=proj
        self.prune=prune
        if self.prune:
            from model.PruneCalculation import PruneHook
            self.pruneHooks=[PruneHook(self.encode_image,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1),
                             PruneHook(self.encoder,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1)]
        else:
            self.pruneHooks=[]
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
        for _,layer in self.encode_image.named_modules():
            if isinstance(layer, nn.ModuleList):
                for block in layer:

                    nn.init.normal_(block.weight, std=1)
                    nn.init.zeros_(block.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1)
                nn.init.zeros_(layer.bias)
        for _,layer in self.encoder.named_modules():
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    nn.init.normal_(block.weight, std=1)
                    nn.init.zeros_(block.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=fc_std)
                nn.init.zeros_(layer.bias)
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
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        return x


    # @torch.jit.script
    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.encode_image(im)
        caption_features1=self.encode_text(captions1)
        caption_features2=self.encode_text(captions2)
        caption_features3=self.encode_text(captions3)
        caption_features4=self.encode_text(captions4)
        caption_features5=self.encode_text(captions5)

        if self.proj=="inv":
            image_features=image_features@ self.text_projection
        elif self.proj=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        else:
            caption_features1=caption_features1@self.text_projection
            caption_features2=caption_features2@self.text_projection
            caption_features3=caption_features3@self.text_projection
            caption_features4=caption_features4@self.text_projection
            caption_features5=caption_features5@self.text_projection
        
        return self.calculate_loss(image_features, caption_features1, caption_features2, caption_features3, caption_features4, caption_features5,norm=self.norm)*self.logit_scale.exp()

    def on_train_epoch_start(self) -> None:
        if self.prune:
            for hook in self.pruneHooks:
                hook.set_up()

    def on_train_epoch_end(self) -> None:
         if self.prune:
            for hook in self.pruneHooks:
                hook.remove()
        
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones(batch[0].shape[0],dtype=torch.float,device=self.device))))))
        
        im,captions= batch[0],batch[1]
        
        logits=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])
        self.log("first logit",logits[0,0,0,0,0,0],enable_graph=False)
        self.log("BAD logit",logits[1,2,3,4,5,0],enable_graph=False)

        lossim = self.loss(logits, labels)
        loss1 = self.loss(logits.permute(1,2,3,4,5,0), labels)
        loss2 = self.loss(logits.permute(2,3,4,5,0,1), labels)
        loss3 = self.loss(logits.permute(3,4,5,0,1,2), labels)
        loss4 = self.loss(logits.permute(4,5,0,1,2,3), labels)
        loss5 = self.loss(logits.permute(5,0,1,2,3,4), labels)
        loss = lossim+loss1+loss2+loss3+loss4+loss5
        loss=loss/6
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        
        return {"loss": loss}

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

    def on_validation_epoch_start(self):
        self.log("Mean Projection Value",self.text_projection.mean(),enable_graph=False)

        self.naninfcount=0
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self.model2.eval()
        self._insert_hooks()
        self.IMhsic_matrix0=torch.zeros([],device=self.device)
        self.IMhsic_matrix1=torch.zeros([],device=self.device)
        self.IMhsic_matrix2=torch.zeros([],device=self.device)
        self.CAPhsic_matrix0=torch.zeros([],device=self.device)
        self.CAPhsic_matrix1=torch.zeros([],device=self.device)
        self.CAPhsic_matrix2=torch.zeros([],device=self.device)
        
        self.eval()
        if not hasattr(self,"classifier"):
            self.classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        
        if len(self.features)>0:
            #When I've collected enough features, I train the classifier
            features=torch.nan_to_num(torch.cat(self.features,dim=0)).cpu().numpy()
            labels=torch.cat(self.labels,dim=0).cpu().numpy()
            # print(features.shape)
            # print(labels.shape)
            self.classifier.fit(features, labels)
            #now restart collection.
            self.labels=[]
            self.features=[]
        self.Linearloss=[]

    def validation_step(self,batch,*args):
        #do stock loss here
        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)
       
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        image_features=self.encode_image(batch[0])
        i=image_features.detach().clone().cpu()
        #run through main mode
        if self.current_epoch>0:
            testpred=self.classifier.predict(image_features.cpu().numpy())
            self.Linearloss.append(np.mean(batch[2].cpu().numpy() == testpred))
            self.log('Linearloss', np.mean(self.Linearloss), prog_bar=True,enable_graph=False, rank_zero_only=True)
       
        self.features.append(i)
        self.labels.append(batch[2].cpu())
        self.model2.encode_image(batch[0])# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,torch.nan_to_num(batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
      
        self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,torch.nan_to_num(batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200))
        joint_HSIC=torch.nan_to_num(batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))), nan=0.0,posinf=1,neginf=-2)
        self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 
        ##Now Do Text
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        choice=torch.randint(0,5,(1,)).item()
        print("choice", choice)
        c=batch[1][:,choice]
        c=c.squeeze()

        captions=self.encode_text(c) #run through main mode
        self.model2.encode_text(c)# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,batch_HSIC2(a)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
        self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,batch_HSIC2(a))
        joint_HSIC=torch.nan_to_num(batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))))
        self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 
       
        logitsI,logitsT=self.calculate_lossStock(image_features, captions) 
        lossim = self.loss(logitsI*self.logit_scale.exp(), labels)
        loss1 = self.loss(logitsT*self.logit_scale.exp(), labels)
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.log('val_loss-stock', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        return loss

    def on_validation_epoch_end(self):

        if self.current_epoch>0:
            self.log("liner_acc",np.sum(self.Linearloss), prog_bar=True,enable_graph=False, rank_zero_only=True)

        self.unfreeze()
        self.train()
        self.plot_results("IM","IMHSIC{}.jpg".format(self.current_epoch))
        self.plot_results("CAP","CAPHSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="IMHSIC{}".format(self.current_epoch), images=["IMHSIC{}.jpg".format(self.current_epoch)])        
            self.logger.log_image(key="CAPHSIC{}".format(self.current_epoch), images=["CAPHSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        print(self.naninfcount)
        del self.model2
        if self.prune:
            for hook in self.pruneHooks:
                    global_entropy = hook.retrieve()
                    hook.remove()        

                    # im_scores =map(lambda name, block: prune_Residual_Attention_block(block, global_entropy[name], self.args["prune_eta"]), filter(lambda name,block: isinstance(block, ResidualAttentionBlock) and name in global_entropy.keys(), self.encode_image.named_modules()[:-1]))
                    # for imscoredict in im_scores:
                    #     for (param_to_prune, im_score) in imscoredict.items():
                    #         prune_module(param_to_prune, im_score, self.args)
                    #then purun accordingly 
        
    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        if isinstance(out, tuple):
            out=out[0]       
        if out.shape[0] == self.hparams.train_batch_size:
            self.__store(out,name,model,layer)
        elif out.shape[1] == self.hparams.train_batch_size:
            self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)

    def __store(self,out,name, model,layer):
        X = out.flatten(1)
        X= torch.nan_to_num((X @ X.t()).fill_diagonal_(0))
        if (torch.isnan(X).any() or torch.isinf(X).any()):
            self.naninfcount+=1
        if model == "model1":
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
        # if layer weight is has self.hparams.train_batch_size in shape or layer.weight is None])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules() ]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        
  
    def export(self):
      
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "IMCKA":self.IMhsic_matrix1 / (torch.sqrt(self.IMhsic_matrix0.unsqueeze(1))*torch.sqrt(self.IMhsic_matrix2.unsqueeze(0))),
            "CAPCKA":self.CAPhsic_matrix1 / (torch.sqrt(self.CAPhsic_matrix0.unsqueeze(1))*torch.sqrt(self.CAPhsic_matrix2.unsqueeze(0))),
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     model_name: str,
                     save_path: str = None,
                     title: str = None):
        title =model_name+" HSIC" if title is None else model_name+title
        fig, ax = plt.subplots()
        if model_name=="IM":
            print(self.IMhsic_matrix0) #46 #Comes out inf on val step
            print(self.IMhsic_matrix2) # 110
            t=self.IMhsic_matrix0.unsqueeze(1)*self.IMhsic_matrix2.unsqueeze(0) #46 x 110
        #print(torch.sum(torch.abs(t)==t))
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            print("im1",self.IMhsic_matrix1)
            print("r", r)
            hsic_matrix = torch.div(self.IMhsic_matrix1.squeeze().t(), r)
            print("hsic",hsic_matrix)
        else:
            print(self.CAPhsic_matrix0.shape,self.CAPhsic_matrix2.shape)
            t=self.CAPhsic_matrix0.unsqueeze(1)*self.CAPhsic_matrix2.unsqueeze(0)
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            print("cap1", self.CAPhsic_matrix1.shape)
            print("r",r.shape)
            hsic_matrix = torch.div(self.CAPhsic_matrix1.squeeze().t() , r)
        hsic_matrix=torch.nan_to_num(hsic_matrix,nan=0)
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



def batch_HSIC2(K):
    #K is Layers x B x B
    a=torch.sum(K,dim=-1)
    #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
    b=torch.sum(K,dim=-2)
    c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[-2] - 1),torch.sum(a*b,dim=1),alpha=2)
    #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
    output=torch.add(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
    return torch.div(output,(K.shape[-2]*(K.shape[-2] - 3)))
    #check for why pos infs... 
def batch_HSIC3(K,L):
    K=K.unsqueeze(1) # 46,1,B,B
    L=L.unsqueeze(0) # 1,46, B,B
    a=torch.sum(L,dim=-1) #1,46,10
    b=torch.sum(K,dim=-2) #46,1,10
    #print(a.shape,b.shape)
    c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
    #print(c.shape) # expect LayerK, LayerL, 
    return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
    #returns many pos infs 