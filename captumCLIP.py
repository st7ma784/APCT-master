
import numpy as np
import torch
import torch.nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature
import os
import skimage
from captum.attr import TokenReferenceBase
import clip
default_cmap=LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model=model.cuda().train()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
    
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

token_reference=TokenReferenceBase(0)
decoder=clip.simple_tokenizer.SimpleTokenizer()
print("Torch version:", torch.__version__)
#print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

class clipWrapper(torch.nn.Module):
    def __init__(self, model,output_item=0):
        super(clipWrapper, self).__init__()
        self.model = model
        self.output_item = output_item
    def forward(self, *input):
        return self.model(*input)[self.output_item]
model=clipWrapper(model,output_item=0)


#nd their textual descriptions
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}

# %%
original_images = []
images = []
texts = []
plt.figure(figsize=(16, 5))

for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])

plt.tight_layout()

image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()


def baseline_func(input):
    return torch.zeros(input.shape, device=input.device)




def formatted_data_iter():
    # dataset = torch.utils.data.TensorDataset(image_input)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=default_collate)
    # dataset2 = torch.utils.data.TensorDataset(text_tokens)
    # loader2 = torch.utils.data.DataLoader(dataset2, batch_size=4, shuffle=False, collate_fn=default_collate)
    yield Batch(inputs=(image_input,text_tokens.to(dtype=torch.long)),labels=torch.arange(8))


def baseline_text(x):
    ref_indices = token_reference.generate_reference(x.size(0), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
).unsqueeze(0)
    return model.model.token_embedding(ref_indices).squeeze(0)

def input_text_transform(x):
    return model.model.token_embedding(x)
#model = clip.visual_transformer


# wrap the inputs into layers incase we wish to use a layer method
model = ModelInputWrapper(model)

# `device_ids` contains a list of GPU ids which are used for paralelization supported by `DataParallel`
model = torch.nn.DataParallel(model)
attr = LayerIntegratedGradients(model, [model.visual])

def vqa_resnet_interpret(img, caption, PotentionCaptions):
    # original_image = transforms.Compose([transforms.Scale(int(image_size / central_fraction)),
    #                                transforms.CenterCrop(image_size), transforms.ToTensor()])(img) 
    #q_len = get index of highest token in caption 
    q_len=torch.argmax(caption,dim=-1)+1
    # generate reference for each sample
    q_reference_indices = token_reference.generate_reference(q_len).unsqueeze(0)
        
    inputs = (img, caption.unsqueeze(0))
    baselines = (img * 0.0, q_reference_indices)
    
        
    outputs = model(*inputs)
        
    # Make a prediction. The output of this prediction will be visualized later.
    pred, answer_idx = F.softmax(outputs, dim=1).data.cpu().max(dim=1)

    attributions = attr.attribute(inputs=inputs,
                                baselines=baselines,
                                target=answer_idx,
                                #additional_forward_args=q_len.unsqueeze(0),
                                n_steps=30)
        
    # Visualize text attributions
    text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()
    for target in PotentionCaptions:
        
        vis_data_records = [visualization.VisualizationDataRecord(
                                attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm,
                                pred[0].item(),
                                PotentionCaptions[ answer_idx ],
                                PotentionCaptions[ answer_idx ],
                                target,
                                attributions[1].sum(),       
                                caption,
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
