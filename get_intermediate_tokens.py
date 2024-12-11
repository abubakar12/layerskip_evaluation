from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Load the LLaMA model and tokenizer
model_name = "NousResearch/Llama-2-13b-hf"  # Replace with your model
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", 
#                                 token="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",
#                                 cache_dir="/mnt/data/")
# model = AutoModelForCausalLM.from_pretrained(model_name, 
#                                              output_hidden_states=True,
#                                              token="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")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             device_map="auto",
                                             use_safetensors=True,
                                             torch_dtype=torch.float16,
                                             cache_dir="/mnt/data/",
                                             token="")
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir="/mnt/data/",
                                          token="")

# Load the dataset from the text file
with open("dataset_example.txt", "r") as file:
    questions = file.readlines()

# Create a DataFrame from the questions
df = pd.DataFrame(questions, columns=["question"])
df["res"] = "Using Answer Using one word only :  " + df['question'] +  " Answer: "
# df_sampled = df.sample(n=20, random_state=42).reset_index(drop=True)
# print("Number of samples: ", len(df_sampled))
df_sampled = df.copy()
# Initialize columns for each layer
num_layers = 0
layer_columns = []

# Process each row and compute the tokens for all layers
results = []
for _, row in tqdm(df_sampled.iterrows()):
    input_text = row["res"]
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Forward pass to get all hidden states9Ysma17BzRjErRT
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of hidden states for each layer
    
    # Number of layers
    if num_layers == 0:
        num_layers = len(hidden_states)
        layer_columns = [f"layer_{i}" for i in range(num_layers, 0, -1)]  # Reverse order
    
    # Compute top token for each layer
    layer_results = []
    for layer_hidden_state in reversed(hidden_states):  # Iterate from last to first layer
        # layer_hidden_state = layer_hidden_state.unsqueeze(0).unsqueeze(0)
        
        # Add normalization before unembedding
        normalized_hidden_state = model.model.norm(layer_hidden_state)
        logits = model.lm_head(normalized_hidden_state)  # Project to vocab size

        probs = F.softmax(logits, dim=-1)  # Compute probabilities
        
        last_token_probs = probs[:, -1, :]  # Select the last token probabilities
        top_token_index = torch.argmax(last_token_probs, dim=-1).item()
        top_token = tokenizer.convert_ids_to_tokens([top_token_index])[0]
        
        layer_results.append(top_token)
    
    # Append the results for this row
    results.append([row["res"]] + layer_results)

# Create a new DataFrame with the required structure
columns = ["Res"] + layer_columns
final_df = pd.DataFrame(results, columns=columns)

# Save the DataFrame to a CSV file
final_df.to_csv("output_results_2_13b_full_model.csv", index=False)  # Specify the desired filename

print(final_df.head())
