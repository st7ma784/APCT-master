
import numpy as np
import torch

import torch.nn
import torchvision.transforms as transforms


from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from captum.attr import TokenReferenceBase
import clip


def vqa_resnet_interpret(image_filename, caption, PotentionCaptions):
    img = Image.open(image_filename).convert('RGB')
    # original_image = transforms.Compose([transforms.Scale(int(image_size / central_fraction)),
    #                                transforms.CenterCrop(image_size), transforms.ToTensor()])(img) 
    
    image_features = image_to_features(img).requires_grad_().to(device)
    q, q_len = encode_question(caption)
    
    # generate reference for each sample
    q_reference_indices = token_reference.generate_reference(q_len.item(), device=device).unsqueeze(0)

    inputs = (q.unsqueeze(0), q_len.unsqueeze(0))
        
    inputs = (image_features, q.unsqueeze(0))
    baselines = (image_features * 0.0, q_reference_indices)
    
        
    ans = vqa_resnet(*inputs, q_len.unsqueeze(0))
        
    # Make a prediction. The output of this prediction will be visualized later.
    pred, answer_idx = F.softmax(ans, dim=1).data.cpu().max(dim=1)

    attributions = attr.attribute(inputs=inputs,
                                baselines=baselines,
                                target=answer_idx,
                                additional_forward_args=q_len.unsqueeze(0),
                                n_steps=30)
        
    # Visualize text attributions
    text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()
    for target in PotentionCaptions:
        
        vis_data_records = [visualization.VisualizationDataRecord(
                                attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm,
                                pred[0].item(),
                                answer_words[ answer_idx ],
                                answer_words[ answer_idx ],
                                target,
                                attributions[1].sum(),       
                                question.split(),
                                0.0)]
        visualization.visualize_text(vis_data_records)

        # visualize image attributions
        original_im_mat = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
        attributions_img = np.transpose(attributions[0].squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
        visualization.visualize_image_attr_multiple(attributions_img, original_im_mat, 
                                                    ["original_image", "heat_map"], ["all", "absolute_value"], 
                                                    titles=["Original Image", "Attribution Magnitude"],
                                                    cmap=default_cmap,
                                                    show_colorbar=True)
        print('Text Contributions: ', attributions[1].sum().item())
        print('Image Contributions: ', attributions[0].sum().item())
        print('Total Contribution: ', attributions[0].sum().item() + attributions[1].sum().item())