import torch
from transformers import AutoModel, AutoTokenizer
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
os.environ['HUGGINGFACE_TOKEN']=""
from tqdm import tqdm
# Configuration
MODEL_NAME = "facebook/layerskip-llama2-13B"  # Replace with your desired Hugging Face model
MODEL_NAME2 = "meta-llama/Llama-2-13b-hf"
THRESHOLDS = [ 0.1,0.3,0.5,0.9]

# Load both models and tokenizers
model1 = AutoModel.from_pretrained(MODEL_NAME,device_map="auto", token=os.environ['HUGGINGFACE_TOKEN'],cache_dir="/mnt/data")
model2 = AutoModel.from_pretrained(MODEL_NAME2,device_map="auto", token=os.environ['HUGGINGFACE_TOKEN'],cache_dir="/mnt/data")
tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ['HUGGINGFACE_TOKEN'],cache_dir="/mnt/data")
tokenizer2 = tokenizer1
def extract_layer_idx(name):
    """
    Extract a layer index from the module name.
    This function may need adjustment depending on the model's naming conventions.
    For LLaMA-based models, layers often appear as "model.model.layers.X..."
    We'll try to parse out 'layers.X' to find the layer index.
    """
    parts = name.split('.')
    # Common pattern: model.model.layers.<layer_idx>...
    # Let's try to find 'layers' and the next part as index
    for i, p in enumerate(parts):
        if p == 'layers':
            # The next element should be the layer index
            if i+1 < len(parts) and parts[i+1].isdigit():
                return int(parts[i+1])
    return None  # If no layer index found

def get_activation_stats(model, inputs, threshold):
    activation_stats = {}

    def hook(module, input, output):
        activated_neurons = (output > threshold).sum().item()
        total_neurons = output.numel()

        # The module name will be captured via closure (we'll rely on the name from outside scope)
        mod_name = current_module_name
        layer_idx = extract_layer_idx(mod_name)
        
        if layer_idx is None:
            # Place them in a separate key if layer_idx is not found
            layer_idx = 'other'

        if layer_idx not in activation_stats:
            activation_stats[layer_idx] = {}

        if mod_name not in activation_stats[layer_idx]:
            activation_stats[layer_idx][mod_name] = {
                "activated_neurons": 0,
                "total_neurons": 0
            }

        activation_stats[layer_idx][mod_name]["activated_neurons"] += activated_neurons
        activation_stats[layer_idx][mod_name]["total_neurons"] += total_neurons

    hooks = []
    # To preserve the name in the hook, we can use a closure trick:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Create a local reference to 'name' for the hook
            def make_hook(name):
                def local_hook(module, input, output):
                    nonlocal name
                    activated_neurons = (output > threshold).sum().item()
                    total_neurons = output.numel()
                    layer_idx = extract_layer_idx(name)
                    if layer_idx is None:
                        layer_idx = 'other'
                    if layer_idx not in activation_stats:
                        activation_stats[layer_idx] = {}
                    if name not in activation_stats[layer_idx]:
                        activation_stats[layer_idx][name] = {
                            "activated_neurons": 0,
                            "total_neurons": 0
                        }
                    activation_stats[layer_idx][name]["activated_neurons"] += activated_neurons
                    activation_stats[layer_idx][name]["total_neurons"] += total_neurons
                return local_hook

            hook_fn = make_hook(name)
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass to compute activations
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activation_stats

# Load the dataset
with open("dataset_example.txt", "r") as file:
    questions = file.readlines()

# Create a DataFrame from the questions
df = pd.DataFrame(questions, columns=["question"])
df["res"] = "Using Answer Using one word only :  " + df['question'] +  " Answer: "
# df=df.head(200)
# Modify the analysis section to handle multiple sentences
def analyze_multiple_sentences(model, tokenizer, sentences, threshold):
    all_stats = []
    for sentence in tqdm(sentences):
        inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
        stats = get_activation_stats(model, inputs, threshold)
        all_stats.append(stats)
    
    # Compute average statistics
    avg_stats = {}
    for layer_idx in all_stats[0].keys():
        if layer_idx == 'other':
            continue
        
        avg_stats[layer_idx] = {}
        for module_name in all_stats[0][layer_idx].keys():
            avg_stats[layer_idx][module_name] = {
                "activated_neurons": np.mean([stats[layer_idx][module_name]["activated_neurons"] for stats in all_stats]),
                "total_neurons": all_stats[0][layer_idx][module_name]["total_neurons"]  # Total neurons remain constant
            }
    
    return avg_stats

# Modify the analysis section to loop through thresholds
for threshold in THRESHOLDS:
    print(f"\nAnalyzing with threshold: {threshold}")
    activation_stats1 = analyze_multiple_sentences(model1, tokenizer1, df["res"].tolist(), threshold)
    activation_stats2 = analyze_multiple_sentences(model2, tokenizer2, df["res"].tolist(), threshold)
    
    # Modified printing and visualization section
    def plot_activation_statistics(activation_stats1, activation_stats2, threshold):
        plt.figure(figsize=(3, 2), dpi=300)
        
        def plot_percentages(activation_stats, color, label):
            layer_indices = []
            activation_percentages = []
            
            for layer_idx in sorted(activation_stats.keys()):
                if layer_idx == 'other':
                    continue
                    
                layer_stats = activation_stats[layer_idx]
                total_activated = sum(stats["activated_neurons"] for stats in layer_stats.values())
                total_neurons = sum(stats["total_neurons"] for stats in layer_stats.values())
                
                if total_neurons > 0:
                    activation_percentage = (total_activated / total_neurons) * 100
                    layer_indices.append(layer_idx)
                    activation_percentages.append(activation_percentage)

            plt.plot(layer_indices, activation_percentages, marker='o', linewidth=1.5, 
                    markersize=3, label=label, color=color, alpha=0.8)
        
        # Plot percentages for both models
        plot_percentages(activation_stats1, 'blue', "LayerSkip LLaMA")
        plot_percentages(activation_stats2, 'red', "Original LLaMA")
        
        plt.title('Activation Percentage Comparison', fontsize=8)
        plt.xlabel('Layer Number', fontsize=7)
        plt.ylabel('Activation Percentage (%)', fontsize=7)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=6)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        
        plt.tight_layout()
        plt.savefig(f'model_{MODEL_NAME2[11:]}_comparison_percentage_threshold_{threshold:.1f}.png', 
                    dpi=300, bbox_inches='tight', format='png')
        plt.close()

    # Call plotting function with threshold
    plot_activation_statistics(activation_stats1, activation_stats2, threshold)
