

import torch
from functools import partial,reduce

def calculate_lossStock(I, C1):

    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    logits_per_text =  C1 @ I.T
    #calculate loss
    return logits_per_image, logits_per_text
def calculate_loss(  I, C1, C2, C3, C4, C5,norm=True):
    if norm:
        I = I / I.norm(dim=-1, keepdim=True)
        C1 = C1 / C1.norm(dim=-1, keepdim=True)
        C2 = C2 / C2.norm(dim=-1, keepdim=True)
        C3 = C3 / C3.norm(dim=-1, keepdim=True)
        C4 = C4 / C4.norm(dim=-1, keepdim=True)
        C5 = C5 / C5.norm(dim=-1, keepdim=True) 
    return torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",I,C1,C2),torch.einsum("az,bz,cz->abcz",C3,C4,C5))
def calculate_loss2(  I, C1, C2, C3, C4, C5,norm=True):
    if norm:
        I = I / I.norm(dim=-1, keepdim=True)
        C1 = C1 / C1.norm(dim=-1, keepdim=True)
        C2 = C2 / C2.norm(dim=-1, keepdim=True)
        C3 = C3 / C3.norm(dim=-1, keepdim=True)
        C4 = C4 / C4.norm(dim=-1, keepdim=True)
        C5 = C5 / C5.norm(dim=-1, keepdim=True)

    #1- sum sqrt(sum((x_i - mean(x))^2))
    return 1-torch.sum(torch.sqrt(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6)),dim=-1)
def calculate_loss3( I, C1, C2, C3, C4, C5,norm=True):
    if norm:
        I = I / I.norm(dim=-1, keepdim=True)
        C1 = C1 / C1.norm(dim=-1, keepdim=True)
        C2 = C2 / C2.norm(dim=-1, keepdim=True)
        C3 = C3 / C3.norm(dim=-1, keepdim=True)
        C4 = C4 / C4.norm(dim=-1, keepdim=True)
        C5 = C5 / C5.norm(dim=-1, keepdim=True)
    return  1-torch.sqrt(torch.sum(torch.pow(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6),2),dim=-1))


def calculate_loss4(I, C1, C2, C3, C4, C5,norm=True):
    if norm:
        I = I / I.norm(dim=-1, keepdim=True)
        C1 = C1 / C1.norm(dim=-1, keepdim=True)
        C2 = C2 / C2.norm(dim=-1, keepdim=True)
        C3 = C3 / C3.norm(dim=-1, keepdim=True)
        C4 = C4 / C4.norm(dim=-1, keepdim=True)
        C5 = C5 / C5.norm(dim=-1, keepdim=True)

    return torch.sum(torch.sqrt(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                                torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                                torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)])),dim=-1)

def calculate_loss5(I, C1, C2, C3, C4, C5,norm=True):
    if norm:
        I = I / I.norm(dim=-1, keepdim=True)
        C1 = C1 / C1.norm(dim=-1, keepdim=True)
        C2 = C2 / C2.norm(dim=-1, keepdim=True)
        C3 = C3 / C3.norm(dim=-1, keepdim=True)
        C4 = C4 / C4.norm(dim=-1, keepdim=True)
        C5 = C5 / C5.norm(dim=-1, keepdim=True)
    return torch.sum(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6),dim=-1)

def calculate_loss6(I, C1, C2, C3, C4, C5,norm=True):
        if norm:
            I = I / I.norm(dim=-1, keepdim=True)
            C1 = C1 / C1.norm(dim=-1, keepdim=True)
            C2 = C2 / C2.norm(dim=-1, keepdim=True)
            C3 = C3 / C3.norm(dim=-1, keepdim=True)
            C4 = C4 / C4.norm(dim=-1, keepdim=True)
            C5 = C5 / C5.norm(dim=-1, keepdim=True)

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
    # @torch.jit.script
