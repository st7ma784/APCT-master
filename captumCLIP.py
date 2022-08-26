
from email.policy import default
import numpy as np
import torch

print("Torch version:", torch.__version__)

import clip
clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# %% [markdown]
# # Image Preprocessing
# 
# We resize the input images and center-crop them to conform with the image resolution that the model expects. Before doing so, we will normalize the pixel intensity using the dataset mean and standard deviation.
# 
# The second return value from `clip.load()` contains a torchvision `Transform` that performs this preprocessing.
# 
# 

# %% [markdown]
# # Text Preprocessing
# 
# We use a case-insensitive tokenizer, which can be invoked using `clip.tokenize()`. By default, the outputs are padded to become 77 tokens long, which is what the CLIP models expects.

# %%

# %% [markdown]
# # Setting up input images and texts
# 
# We are going to feed 8 example images and their textual descriptions to the model, and compare the similarity between the corresponding features.
# 
# The tokenizer is case-insensitive, and we can freely give any suitable textual descriptions.

# %%
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
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


# %% [markdown]
# ## Building features
# 
# We normalize the images, tokenize each text input, and run the forward pass of the model to get the image and text features.

# %%
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

# %%
image_features = model.encode_image(image_input).float()
text_features = model.encode_text(text_tokens).float()

# %% [markdown]
# ## Calculating cosine similarity
# 
# We normalize the features and calculate the dot product of each pair.

# %%
with torch.no_grad():
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T



# %%
count = len(descriptions)

plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
# plt.colorbar()
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
  plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=20)

# %%
print(text_tokens.shape)

# %%
model(image_input,text_tokens)[1].shape

# %%
import os

import torch
import torch.nn
import torchvision.transforms as transforms


from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature
def get_classes():
    return texts

def baseline_func(input):
    return input * 0

from torch.utils.data import default_collate

def formatted_data_iter():
    # dataset = torch.utils.data.TensorDataset(image_input)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=default_collate)
    # dataset2 = torch.utils.data.TensorDataset(text_tokens)
    # loader2 = torch.utils.data.DataLoader(dataset2, batch_size=4, shuffle=False, collate_fn=default_collate)
    yield Batch(inputs=(image_input,text_tokens.to(dtype=torch.long)),labels=torch.arange(8))
    
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
from captum.attr import TokenReferenceBase, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

token_reference=TokenReferenceBase(0)
def baseline_text(x):
    ref_indices = token_reference.generate_reference(x.size(0), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
).unsqueeze(0)
    return model.token_embedding(ref_indices).squeeze(0)

def input_text_transform(x):
    return model.token_embedding(x)
decoder=clip.simple_tokenizer.SimpleTokenizer()
#model = clip.visual_transformer
visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: o,
    classes=get_classes(),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        ),
        TextFeature(
        "Question",
        input_transforms=[], #input_text_transform
        baseline_transforms=[baseline_text],
        visualization_transform=decoder.decode, #detokenize
    ),
    ],
    dataset=formatted_data_iter(),
)


visualizer.render()

visualizer.visualize()

