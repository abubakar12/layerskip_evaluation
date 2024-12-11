import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import os
os.environ['HUGGINGFACE_TOKEN']=""
# Load a pre-trained model and tokenizer
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_NAME = "facebook/layerskip-llama3.2-1B"  # Replace with your desired Hugging Face model
MODEL_NAME2 = "meta-llama/Llama-3.2-1B"
THRESHOLDS = [ 0.1,0.3,0.5,0.9]

# Load both models and tokenizers
model = AutoModel.from_pretrained(MODEL_NAME2,device_map="auto", token=os.environ['HUGGINGFACE_TOKEN'],cache_dir="/mnt/data")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ['HUGGINGFACE_TOKEN'],cache_dir="/mnt/data")


# Input text
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Forward pass to get attention weights
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # List of (num_layers, batch_size, num_heads, seq_len, seq_len)

# Extract number of layers and heads
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = attentions[0].shape[-1]

# Calculate grid dimensions based on number of heads
grid_size = int(np.ceil(np.sqrt(num_heads)))

# Save individual layer plots
for layer in range(num_layers):
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Layer {layer} Attention Heads", fontsize=16)
    
    for head in range(num_heads):
        plt.subplot(grid_size, grid_size, head + 1)
        attention_weights = attentions[layer][0, head].detach().cpu().numpy()
        im = plt.imshow(attention_weights, cmap="viridis")
        plt.title(f"Head {head}", fontsize=12)
        plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig(f'attention_heatmaps_layer_{layer}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Plot last 5 layers together (one figure per layer)
for layer in range(num_layers - 5, num_layers):
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Layer {layer} Attention Heads", fontsize=16)
    
    for head in range(num_heads):
        plt.subplot(grid_size, grid_size, head + 1)
        attention_weights = attentions[layer][0, head].detach().cpu().numpy()
        im = plt.imshow(attention_weights, cmap="viridis")
        plt.title(f"Head {head}", fontsize=12)
        plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig(f'attention_heatmaps_layer_{layer}_last5.png', bbox_inches='tight', dpi=300)
    plt.close()
