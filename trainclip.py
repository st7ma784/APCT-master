
import pytorch_lightning
from transformers import AutoModelWithLMHead, get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from typing import Union,Tuple,Optional
from clip.model import CLIP,Transformer,LayerNorm
from clip.simple_tokenizer import SimpleTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint,TQDMProgressBar
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
    def forward(self, query, response):
        image_features = self.encode_query(query)
        text_features = self.encode_response(response)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
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
                learning_rate: float = 2e-4,
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 32,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                **kwargs,
                ):

        super(LightningCLIPModule, self).__init__()
        self.save_hyperparameters()
        self.clip = myclip(embed_dim= 512,
                 context_length= 77,
                 vocab_size= 32100,
                 transformer_width= 512,
                 transformer_heads= 32,
                 transformer_layers= 4)
        self.clip.dtype=self.dtype
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.lossq = torch.nn.CrossEntropyLoss()
        self.lossr = torch.nn.CrossEntropyLoss()

    def forward(self, query, response):
        return self.clip(query, response)
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        labels=torch.arange(batch[0].shape[0],dtype=torch.long).to(self.device,non_blocking=True)
        query ,response= batch[0].squeeze(1),batch[1].squeeze(1)
        qlogits, rlogits = self(query, response)
        lossq = self.lossq(rlogits, labels)
        lossr = self.lossr(qlogits, labels)
        loss = lossq+lossr
        loss = loss.mean()
        self.log('train_loss', loss.item())
        return {"loss": loss, "log": {"train_loss": loss}}

            
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.clip.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.clip.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]

        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)
        return [optimizer], [scheduler]
        

if __name__ == "__main__":
    #Load Data Module and begin training
    from BuildSpainDataSet import DataModule
    data_module = DataModule(batch_size=256)
    callbacks=[
        ModelCheckpoint(filename="CLIPModule",save_last=True, every_n_epochs=50, save_on_train_epoch_end=None),
        TQDMProgressBar()
    ]
    trainer=pytorch_lightning.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=100,
        callbacks=callbacks,
        gradient_clip_val=0.25,
        fast_dev_run=False
    )
    model=LightningCLIPModule()
    trainer.fit(model,data_module)