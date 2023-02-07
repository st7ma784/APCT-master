
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer, QuickGELU
from functools import partial,reduce
import clip
from operator import iadd
#add APCT to path
import sys
sys.path.append("./APCT-master")
from warnings import warn
import matplotlib.pyplot as plt
from CKA_test import add_colorbar 
from sklearn.linear_model import LogisticRegression
from core.pattern import PruneHook, set_gamma
from torch.nn.utils.prune import l1_unstructured, random_structured, ln_structured, remove, identity, is_pruned

 
class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                useclip_en='v40',
                useclip_im=False,
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
        #print("learning_rate",learning_rate)
        self.args=kwargs
        self.args["prune_eta"] = -1
        self.args["method"] = "Hard"

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
        self.model_hookI = PruneHook(self.encode_image,[-1,0,1], 0.1, **self.args)
        self.model_hookT = PruneHook(self.encoder,[-1,0,1], 0.1, **self.args)
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.lossim=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss1=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss2=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss3=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss4=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss5=torch.nn.CrossEntropyLoss(reduction='mean')
        self.vocab_size = vocab_size

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.initialize_parameters()
        self.handles=[]
        self.features=[]
        
        self.labels=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        print("done")

    def on_train_epoch_start(self) -> None:
        self.model_hookI.set_up()
        self.model_hookT.set_up()

    # def training_step(self, batch, batch_idx):
    #     super().training_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self.model_hookI.remove()
        self.model_hookT.remove()


    
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
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def calculate_loss3(self, I, C1, C2, C3, C4, C5):
  
        return 1-torch.sqrt(torch.sum(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                  torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                                  torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                  torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                                  torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                  torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)]).sub_(
                            torch.pow(reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                                                        C1.view(1,C1.shape[0],1,1,1,1,-1),
                                                        C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                        C3.view(1,1,1,C3.shape[0],1,1,-1),
                                                        C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                        C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6),dim=-1))
    # rewrite calculate_loss3 to accept a list of args as *args 
    def calculate_lossn(self, *args):
        argdict = dict(enumerate(args))
        ones=torch.ones(len(args),len(args))
        ones=ones.scatter(0,torch.arange(len(args)),torch.tensor([i.shape[0] for i in args]))
        termlist=[arg.view(ones[i]) for i,arg in argdict.items()] # does this really need to be ordered?
        term1=reduce(torch.add,[torch.pow(i,2) for i in termlist])
        term2=reduce(torch.add,termlist)
        return 1-torch.sqrt(torch.sum(term1.sub_(term2, alpha=1/len(args)), dim=-1))

    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.encode_image(im)
        #self.features.append(image_features.clone().detach().cpu())
        #image_features=image_features/ torch.norm(image_features, dim=1, keepdim=True)
        caption_features1=self.encode_text(captions1)
        #caption_features1=caption_features1/ torch.norm(caption_features1, dim=1, keepdim=True)
        caption_features2=self.encode_text(captions2)
        #caption_features2=caption_features2/ torch.norm(caption_features2, dim=1, keepdim=True)
        caption_features3=self.encode_text(captions3)
        #caption_features3=caption_features3/ torch.norm(caption_features3, dim=1, keepdim=True)
        caption_features4=self.encode_text(captions4)
        #caption_features4=caption_features4/ torch.norm(caption_features4, dim=1, keepdim=True)
        caption_features5=self.encode_text(captions5)
        #caption_features5=caption_features5/ torch.norm(caption_features5, dim=1, keepdim=True)

        return self.calculate_loss3(image_features, caption_features1, caption_features2, caption_features3, caption_features4, caption_features5)*self.logit_scale.exp()

        


    def training_step(self, batch, batch_idx,optimizer_idx=0):
        labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.lossim.ignore_index))))
        labels=labels+self.lossim.ignore_index
        
        im,captions= batch[0],batch[1]
        
        logits=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])
        
        lossim = self.lossim(logits, labels)

        loss1 = self.loss1(logits.permute(1,2,3,4,5,0), labels)
        loss2 = self.loss2(logits.permute(2,3,4,5,0,1), labels)
        loss3 = self.loss3(logits.permute(3,4,5,0,1,2), labels)
        loss4 = self.loss4(logits.permute(4,5,0,1,2,3), labels)
        loss5 = self.loss5(logits.permute(5,0,1,2,3,4), labels)
        loss = lossim+loss1+loss2+loss3+loss4+loss5
        loss=loss/6
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        return loss

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

    def batch_HSIC2(self,K):
        #K is Layers x B x B
        a=torch.sum(K,dim=-1)
        #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
        b=torch.sum(K,dim=-2)
        c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[-2] - 1),torch.sum(a*b,dim=1),alpha=2)
        #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
        output=torch.add(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
        return torch.div(output,(K.shape[-2]*(K.shape[-2] - 3)))
        #check for why pos infs... 
    def batch_HSIC3(self,K,L):
        K=K.unsqueeze(1) # 46,1,B,B
        L=L.unsqueeze(0) # 1,46, B,B
        a=torch.sum(L,dim=-1) #1,46,10
        b=torch.sum(K,dim=-2) #46,1,10
        #print(a.shape,b.shape)
        c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
        #print(c.shape) # expect LayerK, LayerL, 
        return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
        #returns many pos infs 
    def on_validation_epoch_start(self):
        
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

    def calculate_lossStock(self, I, C1):
  
        #normalize image and text features
        I = I / I.norm(dim=-1, keepdim=True)
        C1 = C1 / C1.norm(dim=-1, keepdim=True)
        #calculate logits
        logits_per_image = I @ C1.T
        logits_per_text = C1 @ I.T
        #calculate loss
        return logits_per_image*self.logit_scale.exp(), logits_per_text*self.logit_scale.exp()
    def validation_step(self,batch,*args):
        
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        image_features=self.encode_image(batch[0])
        i=image_features.cpu() #run through main mode
        if self.current_epoch>0:
            testpred=self.classifier.predict(i.numpy())
            self.Linearloss.append(np.mean(batch[2].cpu().numpy() == testpred))
            self.log('Linearloss', np.mean(self.Linearloss), prog_bar=True,enable_graph=False, rank_zero_only=True)
       
        self.features.append(i)
        self.labels.append(batch[2].cpu())
        self.model2.encode_image(batch[0])# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,torch.nan_to_num(self.batch_HSIC2(a),nan=0.0,posinf=1,neginf=-2)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
      
        self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,torch.nan_to_num(self.batch_HSIC2(a),nan=0.0,posinf=1,neginf=-2))
        joint_HSIC=torch.nan_to_num(self.batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))), nan=0.0,posinf=1,neginf=-2)
        self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 
        ##Now Do Text
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        c=batch[1][:,torch.randint(0,5,(1,))]
        c=c.squeeze()
        print(c.shape)
        captions=self.encode_text(c) #run through main mode
        self.model2.encode_text(c)# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
        self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,self.batch_HSIC2(a))
        #joint=torch.nan_to_num(self.batch_HSIC3(a,torch.stack(list(self.model1_features.values()))))
        joint_HSIC=torch.nan_to_num(self.batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))))
        #print(joint_HSIC)
        # if not hasattr(self,'CAPhsic_matrix1'):
        #     self.CAPhsic_matrix1=torch.zeros(joint_HSIC.shape,device=self.device)
        self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 
        #Just do the classification loss on Cifar100
        if self.current_epoch>0:
            testpred=self.classifier.predict(i.numpy())
            self.Linearloss.append(np.mean(batch[2].cpu().numpy() == testpred))
            self.log('Linearloss', np.mean(self.Linearloss), prog_bar=True,enable_graph=False, rank_zero_only=True)
            return {"loss":np.mean(self.Linearloss)}
        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)

        logitsI,logitsT=self.calculate_lossStock(image_features, captions)
        lossim = self.lossim(logitsI, labels)
        loss1 = self.loss1(logitsT, labels)
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.log('val_loss-stock', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):


        # Evaluate using the logistic regression classifier
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
        
        global_entropy = self.model_hookI.retrieve()
        

        # im_scores =map(lambda name, block: prune_Residual_Attention_block(block, global_entropy[name], self.args["prune_eta"]), filter(lambda name,block: isinstance(block, ResidualAttentionBlock) and name in global_entropy.keys(), self.encode_image.named_modules()[:-1]))
        # for imscoredict in im_scores:
        #     for (param_to_prune, im_score) in imscoredict.items():
        #         prune_module(param_to_prune, im_score, self.args)
        #then purun accordingly 
        self.model_hookI.remove()


        global_entropy = self.model_hookT.retrieve()
        # im_scores =[prune_Residual_Attention_block(block, global_entropy[name], self.args["prune_eta"]) for name, block in [(k,v) for k,v in self.encoder.named_modules()][:-1] if isinstance(block, ResidualAttentionBlock) and name in global_entropy.keys()]
        # for imscoredict in im_scores:
        #     for (param_to_prune, im_score) in imscoredict.items():
        #         prune_module(param_to_prune, im_score, self.args)
        #then purun accordingly 
        self.model_hookT.remove()
     
    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        if isinstance(out, tuple):
            out=out[0]       
            # print("permuted")
        if out.shape[0] == self.hparams.train_batch_size:
            self.__store(out,name,model,layer)
            
        elif out.shape[1] == self.hparams.train_batch_size:
            self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)
        # else:
        #     self.__store(torch.zeros((self.hparams.train_batch_size,out.shape[1]),device=self.device),name,model,layer)
    def __store(self,out,name, model,layer):
        X = out.flatten(1)
        X= torch.nan_to_num((X @ X.t()).fill_diagonal_(0))
        if (torch.isnan(X).any() or torch.isinf(X).any()):
            self.naninfcount+=1
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
        # if layer weight is has self.hparams.train_batch_size in shape or layer.weight is None])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()]) 
        
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules() ]) 
        
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()]) 
        
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        

    def export(self):
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
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

from core.pattern import EntropyHook
from functools import partial
from random import random
from collections import defaultdict











class PruneHook(EntropyHook):
    def __init__(self, model, Gamma, ratio=1, **kwargs):
        super().__init__(model, Gamma, ratio)
        self.device="cuda"
        self.kwargs=kwargs
        self.activations =set([nn.Linear])#set([nn.LeakyReLU, nn.ReLU, nn.ELU, nn.Sigmoid, nn.GELU,QuickGELU, nn.Tanh, nn.PReLU])
        self.Gamma=torch.tensor(Gamma, dtype=torch.float32,device=self.device)
  
    def set_up(self):
        
        self.remove()
        self.features=defaultdict(lambda: torch.zeros((1,self.Gamma.shape[0]+1), dtype=torch.float32, device=self.device))
        self.handles.extend( [module.register_forward_hook(partial(self.hook, layer_name=module_name)) for module_name, module in self.model.named_modules() if type(module) in self.activations])

    def hook(self, layer, input_var, output_var, layer_name):
        #print(layer.__dir__())
        if random() < self.ratio:
            input=output_var.view(output_var.shape[-1],-1)
            hist=torch.bucketize(input, self.Gamma)# returns index of gamma to each value.
            counts=torch.stack([torch.bincount(hist[i,:],max=self.Gamma.shape[0]+1 ) for i in range(hist.shape[0])])
            self.features[layer_name]= counts.add(self.features[layer_name])
   
    def process_layer_entropy(self,layer):

        layer = layer.reshape(self.Gamma.shape[0]+1, -1)
        layer /= layer.sum(axis=0)
        return torch.sum(-layer*torch.log(1e-8+layer),dim=0) # changed from 0 

    def process_block_entropy(self, blockdict):
        if len(blockdict)==0:
            return torch.zeros(1)
        return {k:self.process_layer(v) for k,v in blockdict.items()}

    def retrieve(self,eta=-1):
        if len(self.features.keys())==0:
            return {}
        entropy=self.process_layer_entropy(self.features) 
        print("entropy",entropy.keys())

        #for block_name, block in self.model.named_modules():
        for module_name, module in filter(lambda item : type(item[1]) in self.activations, self.model.named_modules()):
            im_score = compute_importance(module.weight.detach(), entropy[module_name], eta)
            prune_module(module,module_name, im_score, self.args)
            #now consider pruning the near by batch norm layers 
        
        
        '''
                
            LTWeightsDict={name:layer.weight.detach() for name,layer in block.named_modules() if isinstance(layer,nn.Linear)}
            channel_entropy = block_entropy#.mean(tuple(range(1, num_dim)))   # averaged entropy (out_channels, )
            print("channel_entropy",channel_entropy.shape)
            #lt_im_score = compute_importance(weights, channel_entropy, eta)
            lt_importance_dict={K: compute_importance(V, channel_entropy, eta) for K,V in LTWeightsDict.items()}
            for (param_to_prune, im_score) in lt_importance_dict:
                prune_module(param_to_prune, im_score, self.args)
        
        
        '''

   

def prune_module(layer,name, im_score, args):
    
    cur_param = getattr(layer, name)
    num_dims = cur_param.dim()
    if args.method == 'LnStructured':
        if num_dims > 1:
            ln_structured(layer, name, args.amount, 2, dim=0, importance_scores=im_score.cuda())
        else:
            l1_unstructured(layer, name, args.amount, importance_scores=im_score.cuda())
    elif args.method == 'RandomStructured':
        random_structured(layer, name, args.amount, dim=0)
    elif args.method == 'Hard':
        slc = [slice(None)] * num_dims
        tensor_to_pru = im_score[slc]

        hard_ind = tensor_to_pru[(slice(None, ),) + (0,) * (num_dims - 1)]
        num_filters = torch.sum(hard_ind < args.fc_pru_bound).to(torch.int)
        if num_filters == 0:
            identity(layer, name)
        elif 0 < num_filters < len(tensor_to_pru):
            if num_dims > 1:
                ln_structured(layer, name, int(num_filters), 2, dim=0, importance_scores=im_score.cuda())
            else:
                l1_unstructured(layer, name, int(num_filters), importance_scores=im_score.cuda())
       

def compute_importance(weight, channel_entropy, eta):
    """
    Compute the importance score based on weight and entropy of a channel
    :param weight:  Weight of the module, shape as:
                    ConvBlock: in_channels * out_channels * kernel_size_1 * kernel_size_2
                    LinearBlock: in_channels * out_channels
    :param channel_entropy: The averaged entropy of each channel, shape as in_channels * 1 * (1 * 1)
    :param eta: the importance of entropy in pruning,
                -1:     hard prune without using weight
                0:      prune by weight
                1:      prune by channel_entropy
                2: weight * entropy
                else:   eta * channel_entropy * weight
    :return:    The importance_scores
    """
    print("weight and channel_entropy should have the same number of channels {} {} {} ".format(weight.shape, channel_entropy.shape, channel_entropy.ndim)
    )
    if not weight.shape[0] == channel_entropy.shape[0] and channel_entropy.shape[0] == weight.t().shape[0]:
        weight = weight.t()
        print("Transposing weight")
    assert weight.shape[0] == channel_entropy.shape[0] and channel_entropy.ndim == 1   
    weight = abs(weight)

    if eta == -1:
        importance_scores = weight
    elif eta == 0:
        importance_scores = weight
    elif eta == 2:
        importance_scores = channel_entropy * weight
    elif eta == 3:
        importance_scores =1 / (torch.div(1,channel_entropy) +torch.div(1,weight))
    elif eta == 4:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy * normed_weight
    elif eta == 5:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy + normed_weight
    else:
        raise ValueError()

    return importance_scores

#ENTRY POINT
def prune_Residual_Attention_block(block, block_entropy, eta):
    """
    :param block: RA block to be pruned
    :param block_entropy: entropy of the block output (out_channels * H * W)
    :param eta: hyper parameter.
    :return:
    """
    if block_entropy is None:
        return {}
    print("Pruning Residual Attention Block", block_entropy)

    #weights = getattr(block.LT, 'weight').detach()# in original code, LT is either a linear layer or Conv2d layer
    # weightsDict={"attn":block.attn,
    #             "LN1":block.ln_1, #layer norm
    #             "MLP":block.mlp,    
    #             "MLP_cfc":block.mlp.c_fc, #Linear layer
    #             "MLP_gelu":block.mlp.gelu,
    #             "MLP_c_proj":block.mlp.c_proj, #Linear layer
    #             "LN2":block.ln_2, #layer norm
    #             }
    #LNDict={K:V for K,V in weightsDict.items() if isinstance(V,nn.LayerNorm)}
    #print("block_entropy",block_entropy)
    #if block_entropy is empty tensor

    #block entropy is a list of activations at the norm layers.  each element, is a single value of entropy 
    #num_dim = len(block_entropy.shape)   ####THROWS EERRROR                             # num of dimensions
    
    LTWeightsDict={name:layer.weight.detach() for name,layer in block.named_modules() if isinstance(layer,nn.Linear)}
    channel_entropy = block_entropy#.mean(tuple(range(1, num_dim)))   # averaged entropy (out_channels, )
    print("channel_entropy",channel_entropy.shape)
    #lt_im_score = compute_importance(weights, channel_entropy, eta)
    lt_importance_dict={K: compute_importance(V, channel_entropy, eta) for K,V in LTWeightsDict.items()}
    block_type = 'RABlock'

    linear_im_dict = {
        (K,"weight",block_type):V for K,V in lt_importance_dict.items()}
    #lt_im_score_dict={K: compute_importance(V.weight.detach(), channel_entropy, eta) for K,V in weightsDict.items()}
    #bn_im_score = lt_im_score.mean(dim=tuple(range(1, weights.dim())))
    #bn_im_score_dict={K: V.mean(dim=tuple(range(1, LTWeightsDict[K].dim()))) for K,V in lt_importance_dict.items()}


    # im_dict = {
    #     (block.LT, 'weight', block_type): lt_im_score,
    #     (block.BN, 'weight', block_type): bn_im_score,
    #     (block.BN, 'bias', block_type): bn_im_score
    # }
    # bn_weight_im_dict = {
    #     (K,"weight",block_type):V for K,V in bn_im_score_dict.items()}
    # bn_bias_im_dict = {
    #     (K,"bias",block_type):V for K,V in bn_im_score_dict.items()}
    
    return linear_im_dict
