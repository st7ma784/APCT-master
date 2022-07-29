
import pytorch_lightning
from pytorch_lightning import LightningModule
from transformers import AutoModelWithLMHead
import torch.nn as nn
import torch
from clip import clip

from pytorch_lightning.callbacks import TQDMProgressBar
class LightningCLIPModule(LightningModule):
    def __init__(self,
                learning_rate: float = 2e-4,
                adam_epsilon: float = 1e-8,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.clip,_ = clip.load("ViT-B/32", jit=False, device=self.device)
        self.clip.dtype=self.dtype
        self.decoder=AutoModelWithLMHead.from_pretrained("t5-large")  
        self.lm_head = nn.Linear(self.decoder.config.d_model, self.vocab_size, bias=False)
        self.decoder=self.decoder.decoder
        self.loss=torch.nn.CrossEntropyLoss()

    def forward(self, query):
        
        q_features =self.encode_text(query)
        return self.decoder(q_features).logits


    def training_step(self, batch, batch_idx,optimizer_idx=0):
        
        loss=self.loss(self(batch[0]),batch[0])
        loss = loss.mean()
        return {"loss": loss, "log": {"train_loss": loss}}

            
    def configure_optimizers(self):
        parameters=[
           self.decoder.parameters(),
           self.lm_head.parameters(), 
        ]
        optimizer = torch.optim.Adam(
            parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      
        return [optimizer]
import wandb
def train(config={
        "useclip_im":True,
        "useclip_en":False,
        "batchsize":64,
        "learning_rate":2e-4,
        "precision":16,
    }):
    #Load Data Module and begin training
    from BuildSpainDataSet import TriModal
    with wandb.init( project="T5DecoderTraining", entity="st7ma784", job_type="train", config=config) as run:  
        model=LightningCLIPModule(  learning_rate = config["learning_rate"],
                                    adam_epsilon = 1e-8)
        Dataset=TriModal(dir="MS-COCO-ES",transform=model.preprocess)
        data_module = torch.utils.data.DataLoader(Dataset,batch_size=config["batchsize"],shuffle=True,num_workers=4,pin_memory=True,drop_last=True,prefetch_factor=2)
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
            precision=config["precision"]
        )
        
        
        trainer.fit(model,data_module)

if __name__ == '__main__':
    config={
        "batchsize":512,         #[1,4,8,16,32,64]
        "learning_rate":2e-4,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":16,         #[32,16,'bf16']
    }
    train(config)

