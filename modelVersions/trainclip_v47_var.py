
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

    def calculate_loss(self, I, C1, C2, C3, C4, C5):
        #Calculate loss
        #Loss=1 - sum((values - mean)^2)
        arrMean=torch.add(  torch.div( I,6).view( I.shape[0],1,1,1,1,1,-1),
                            torch.div(C1,6).view(1,C1.shape[0],1,1,1,1,-1)).add(
                torch.add(  torch.div(C2,6).view(1,1,C2.shape[0],1,1,1,-1),
                            torch.div(C3,6).view(1,1,1,C3.shape[0],1,1,-1)).add(
                torch.add(  torch.div(C4,6).view(1,1,1,1,C4.shape[0],1,-1),
                            torch.div(C5,6).view(1,1,1,1,1,C5.shape[0],-1))))
        #Now we have the mean in the final dim shape (B,B,B,B,B,B,512)
        #Normally, we'd do something like Val-mean. However, we do this the other way round for speed, and we can do this because abs(a-b)===abs(b-a)
        #L2normm(Allvalues-mean)
        var= torch.sum(torch.sqrt(torch.add(torch.pow(torch.abs(torch.sub(arrMean, I.view( I.shape[0],1,1,1,1,1,-1))),2),
                                                 torch.pow(torch.abs(torch.sub(arrMean,C1.view(1,C1.shape[0],1,1,1,1,-1))),2)).add(
                                       torch.add(torch.pow(torch.abs(torch.sub(arrMean,C2.view(1,1,C2.shape[0],1,1,1,-1))),2),
                                                 torch.pow(torch.abs(torch.sub(arrMean,C3.view(1,1,1,C3.shape[0],1,1,-1))),2))).add(
                                       torch.add(torch.pow(torch.abs(torch.sub(arrMean,C4.view(1,1,1,1,C4.shape[0],1,-1))),2),
                                                 torch.pow(torch.abs(torch.sub(arrMean,C5.view(1,1,1,1,1,C5.shape[0],-1))),2)))),dim=-1)
        return 1-var
        #print(Arr.shape)
    def calculate_loss2( self, I, C1, C2, C3, C4, C5):
        return 1-torch.sum(torch.sqrt(torch.sub(torch.add(  torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                        torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1)).add(
                                            torch.add(  torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                        torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1)).add(
                                            torch.add(  torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                        torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)))),
                                            torch.pow(torch.add(  I.view( I.shape[0],1,1,1,1,1,-1),
                                                                    C1.view(1,C1.shape[0],1,1,1,1,-1)).add(
                                                        torch.add(  C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                                    C3.view(1,1,1,C3.shape[0],1,1,-1)).add(
                                                        torch.add(  C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)))),2),alpha=1/6)),dim=-1)
                                            
    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.encode_image(im)
        self.features.append(image_features.clone().detach().cpu())
        image_features=image_features/ torch.norm(image_features, dim=1, keepdim=True)
        caption_features1=self.encode_text(captions1)
        caption_features1=caption_features1/ torch.norm(caption_features1, dim=1, keepdim=True)
        caption_features2=self.encode_text(captions2)
        caption_features2=caption_features2/ torch.norm(caption_features2, dim=1, keepdim=True)
        caption_features3=self.encode_text(captions3)
        caption_features3=caption_features3/ torch.norm(caption_features3, dim=1, keepdim=True)
        caption_features4=self.encode_text(captions4)
        caption_features4=caption_features4/ torch.norm(caption_features4, dim=1, keepdim=True)
        caption_features5=self.encode_text(captions5)
        caption_features5=caption_features5/ torch.norm(caption_features5, dim=1, keepdim=True)

        logs=self.logit_scale.exp()
        Loss=self.calculate_loss2(image_features, caption_features1, caption_features2, caption_features3, caption_features4, caption_features5)*logs

        return Loss


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
        return {"loss": loss}

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

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
        
        self.naninfcount=0
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self.model2.eval()
        self._insert_hooks()
        self.eval()
        if not self.hasattr(self,"classifier"):
            self.classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        
        if len(self.features)>0:
            #When I've collected enough features, I train the classifier
            features=torch.nan_to_num(torch.cat(self.features,dim=0)).cpu().numpy()
            labels=torch.cat(self.labels,dim=0).cpu().numpy()
            self.classifier.fit(features, labels)
            #now restart collection.
            self.labels=[]
            self.features=[]
        self.Linearloss=0
        
    def validation_step(self,batch,*args):
        
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        i=self.encode_image(batch[0]) #run through main mode
        try:
            testpred=self.classifier.predict(i.cpu().numpy())
        except:
            testpred=np.zeros(i.shape[0])
        self.features.append(i.cpu())
        self.labels.append(batch[2].cpu())
        accuracy = np.mean((batch[2] == testpred)) * 100.

        self.log("liner_acc", accuracy, prog_bar=True,enable_graph=False, rank_zero_only=True)
        #set linear layers to train mode

        self.model2.encode_image(batch[0])# to compare supervision model
        a=torch.stack(list(self.model1_features.values()))
        if not hasattr(self,'IMhsic_matrix0'):
            self.IMhsic_matrix0=torch.zeros((a.shape[0]),device=self.device)
        self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.stack(list(self.model2_features.values()))
        if not hasattr(self,'IMhsic_matrix2'):
            self.IMhsic_matrix2=torch.zeros((a.shape[0]),device=self.device)
        self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,self.batch_HSIC2(a))
        joint_HSIC=torch.stack(list(map(lambda X: self.batch_HSIC3(a,X),list(self.model1_features.values()))))
        if not hasattr(self,'IMhsic_matrix1'):
            self.IMhsic_matrix1=torch.zeros(joint_HSIC.shape,device=self.device)
        self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 

        ##Now Do Text
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        t=self.encode_text(batch[1][:,0]) #run through main mode
        self.model2.encode_text(batch[1][:,0])# to compare supervision model
        a=torch.stack(list(self.model1_features.values()))
        if not hasattr(self,'CAPhsic_matrix0'):
            self.CAPhsic_matrix0=torch.zeros((a.shape[0]),device=self.device)
        self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.stack(list(self.model2_features.values()))
        if not hasattr(self,'CAPhsic_matrix2'):
            self.CAPhsic_matrix2=torch.zeros((a.shape[0]),device=self.device)
        self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,self.batch_HSIC2(a))
        joint_HSIC=torch.stack(list(map(lambda X: self.batch_HSIC3(a,X),list(self.model1_features.values()))))
        if not hasattr(self,'CAPhsic_matrix1'):
            self.CAPhsic_matrix1=torch.zeros(joint_HSIC.shape,device=self.device)
        self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 
        #Just do the classification loss on Cifar100
       
      

    def on_validation_epoch_end(self):


        # Evaluate using the logistic regression classifier
        
        self.unfreeze()
        self.train()
        self.plot_results("IM","HSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="HSIC{}".format(self.current_epoch), images=["HSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        print(self.naninfcount)
        del self.model2
        del self.IMhsic_matrix0,self.IMhsic_matrix1,self.IMhsic_matrix2
        del self.CAPhsic_matrix1,self.CAPhsic_matrix2,self.CAPhsic_matrix0
        
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
            t=self.IMhsic_matrix0.unsqueeze(1)*self.IMhsic_matrix2.unsqueeze(0)
        #print(torch.sum(torch.abs(t)==t))
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            hsic_matrix = self.IMhsic_matrix1 / r
        else:
            t=self.CAPhsic_matrix0.unsqueeze(1)*self.CAPhsic_matrix2.unsqueeze(0)
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            hsic_matrix = self.CAPhsic_matrix1 / r
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

