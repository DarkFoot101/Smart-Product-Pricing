import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # Added for progress tracking

class TextEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading embedding model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Optimization: Use FP16 if on CUDA to speed up inference
        if self.device == 'cuda':
            self.model = self.model.half()
            
        self.model.eval()

    def get_embeddings(self, texts, batch_size=64): # Increased batch size for speed
        all_embeddings = []
        
        # Added tqdm progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Text"):
            batch = texts[i:i+batch_size]
            
            # Added max_length and truncation to prevent memory errors
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=256, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            # Convert back to float32 before converting to numpy
            all_embeddings.append(embeddings.float().cpu().numpy())
            
        return np.vstack(all_embeddings)