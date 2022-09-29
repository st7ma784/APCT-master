from cmath import pi
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
import torch
from tqdm import tqdm
from torch_cka import CKA
device=torch.device('cuda:1')
model1 = resnet18.to(device,non_blocking=True).eval()  # Or any neural network of your choice
model2 = resnet34.to(device,non_blocking=True).eval()  # Or any neural network of your choice
Dataset=torch.utils.data.TensorDataset(torch.rand(16000,3,227,227))
dataloader = DataLoader(Dataset, 
                         batch_size=50, # according to your device memory
                         shuffle=False,
                         num_workers=8,
                         prefetch_factor=4,
                         pin_memory=True,
                     
                         
                         
                         )  # Don't forget to seed your dataloader


#print(resnet18.T_destination)
cka = CKA(model1, model2,
          model1_name="ResNet18",   # good idea to provide names to avoid confusion
          model2_name="ResNet34",   
          device=device)
cka.model1_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
cka.model2_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]

N = len(cka.model1_layers) if cka.model1_layers is not None else len(list(cka.model1.modules()))
M = len(cka.model2_layers) if cka.model2_layers is not None else len(list(cka.model2.modules()))

def _HSIC( K, L):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        N = K.shape[0]
        result = torch.div(torch.sum(torch.sum(K,dim=0))*torch.sum(torch.sum(L,dim=0)),(N - 1) * (N - 2))
        #print("Shape of res",result.shape) #[]
        result = torch.add(result,torch.trace(K @ L)) 
        #print("Shape of res2",result.shape)#[]
        result = torch.sub(result,torch.mul(torch.sum(K,dim=0)@torch.sum(L,dim=1),2 / (N - 2)))
        #print("Shape of res3",result.shape)#[]
        #result= torch.div(result, (N * (N - 3)))
        #print("Shape of res4",result.shape)#[]
        return result
def _HSIC2( K):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        N = K.shape[0]
        # resulta=torch.div(torch.sum(torch.sum(K,dim=0)),N-1)
        # resultb=torch.div(torch.sum(torch.sum(K,dim=0)),N-2)
        # result=torch.mul(resulta,resultb)
        result = torch.div(torch.pow(torch.sum(torch.sum(K,dim=0)),2),(N - 1) * (N - 2))#or sum(K)/N-1  * sum(K)/N-2
        result = torch.add(result,torch.trace(K@K)) 
        result = torch.sub(result,torch.mul(torch.sum(torch.mul(torch.sum(K,dim=-2),torch.sum(K,dim=-1))),2 / (N - 2)))
    
        return result

def _BHSIC2( K):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        #print(K.shape)

        N = K.shape[1]
        result = torch.div(torch.pow(torch.sum(torch.sum(K,dim=-2),dim=-1),2),(N - 1) * (N - 2))#or sum(K)/N-1  * sum(K)/N-2
        result = torch.add(result,torch.sum(torch.diagonal(torch.matmul(K,K),dim1=1,dim2=2),dim=1)) 
        result = torch.sub(result,torch.mul(torch.sum(torch.mul(torch.sum(K,dim=-2),torch.sum(K,dim=-1)),dim=-1),2 / (N - 2)))
        return result

def _BHSIC( K, L):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        N = K.shape[0]

        resa=torch.div(torch.sum(torch.sum(K,dim=-2),dim=-1,keepdim=True),N-1)
        resb=torch.div(torch.sum(torch.sum(L,dim=-2),dim=-1,keepdim=True),N-2)
        result = resa@resb.t()
        result = torch.add(result,torch.sum(torch.diagonal(torch.matmul(K.unsqueeze(1),L),dim1=-2,dim2=-1),dim=-1)) 
        result= torch.sub(result,torch.mul(torch.sum(K,dim=-2)@torch.sum(L,dim=-1).t(),2 / (N - 2)))

        return result

num_batches = len(dataloader)

for x1 in tqdm(dataloader):
    i=x1[0].to(device,non_blocking=True)
    num_batches = len(dataloader)
    cka.model1_features = {}
    cka.model2_features = {}
    model1(i)
    model2(i)
    features,features2=[],[]
    for _, feat1 in cka.model1_features.items():
        X = feat1.flatten(1)
        features.append((X @ X.t()).fill_diagonal_(0))
    for _,feat2 in cka.model2_features.items():
        Y = feat2.flatten(1)
        features2.append((Y @ Y.t()).fill_diagonal_(0))
    cka.m2_matrix=_BHSIC2(torch.stack(features2,dim=0)).unsqueeze(0)#//(50 * (50 - 3))
    cka.hsic_matrix=_BHSIC2(torch.stack(features,dim=0)).unsqueeze(1)#/(50 * (50 - 3))
    cka.m1_matrix =_BHSIC(torch.stack(features,dim=0),torch.stack(features2,dim=0))#/(50 * (50 - 3))
    cka.res_matrix = torch.div(cka.m1_matrix, torch.sqrt(torch.abs(torch.mul(cka.hsic_matrix,cka.m2_matrix)))+0.000001)
    print(cka.res_matrix)