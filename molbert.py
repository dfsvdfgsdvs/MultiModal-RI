import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict


class SmilesTokenizer:
    def __init__(self, vocab=None, max_length=256, pad_token='<PAD>', 
                 unk_token='<UNK>', cls_token='<CLS>', sep_token='<SEP>', mask_token='<MASK>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.max_length = max_length
        
        self.special_tokens = [pad_token, unk_token, cls_token, sep_token, mask_token]
        
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = defaultdict(lambda: len(self.vocab))
            for token in self.special_tokens:
                _ = self.vocab[token]
        
        self.pad_token_id = self.vocab[pad_token]
        self.unk_token_id = self.vocab[unk_token]
        self.cls_token_id = self.vocab[cls_token]
        self.sep_token_id = self.vocab[sep_token]
        self.mask_token_id = self.vocab[mask_token]
    
    def tokenize(self, smiles):
        tokens = []
        i = 0
        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i:i+2] in ['Cl', 'Br', '@@']:
                tokens.append(smiles[i:i+2])
                i += 2
            elif smiles[i] == '[':
                j = smiles.find(']', i)
                if j != -1:
                    tokens.append(smiles[i:j+1])
                    i = j + 1
                else:
                    tokens.append(smiles[i])
                    i += 1
            else:
                tokens.append(smiles[i])
                i += 1
        return tokens
    
    def encode(self, smiles, add_special_tokens=True, max_length=None, padding=True, truncation=True):
        if max_length is None:
            max_length = self.max_length
        
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.sep_token]
        
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        attention_mask = [1] * len(token_ids)
        
        if padding:
            padding_length = max_length - len(token_ids)
            if padding_length > 0:
                token_ids = token_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
    
    def batch_encode(self, smiles_list, **kwargs):
        batch_input_ids = []
        batch_attention_mask = []
        
        for smiles in smiles_list:
            encoded = self.encode(smiles, **kwargs)
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask
        }
    
    def build_vocab(self, smiles_list):
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            for token in tokens:
                _ = self.vocab[token]
        return dict(self.vocab)
    
    def __len__(self):
        return len(self.vocab)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.w_o(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class MolBERTEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=6, 
                 d_ff=3072, max_length=256, dropout=0.1):
        super(MolBERTEncoder, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask=None, soft_prompt=None):
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        
        if soft_prompt is not None:
            prompt_mask = torch.ones(
                soft_prompt.size(0), soft_prompt.size(1),
                device=soft_prompt.device, dtype=attention_mask.dtype if attention_mask is not None else torch.long
            )
            x = torch.cat([soft_prompt, x], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        x = self.layer_norm(x)
        
        if soft_prompt is not None:
            cls_output = x[:, soft_prompt.size(1), :]
        else:
            cls_output = x[:, 0, :]
        pooled_output = self.pooler(cls_output)
        
        return {
            'last_hidden_state': x,
            'pooler_output': pooled_output
        }


class MolBERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=6,
                 d_ff=3072, max_length=256, dropout=0.1, output_dim=None):
        super(MolBERT, self).__init__()
        
        self.encoder = MolBERTEncoder(vocab_size, d_model, num_heads, num_layers, 
                                       d_ff, max_length, dropout)
        
        self.output_dim = output_dim
        if output_dim is not None:
            self.projection = nn.Linear(d_model, output_dim)
        else:
            self.projection = None
    
    def forward(self, input_ids, attention_mask=None, soft_prompt=None):
        outputs = self.encoder(input_ids, attention_mask, soft_prompt)
        
        if self.projection is not None:
            outputs['projected_output'] = self.projection(outputs['pooler_output'])
        
        return outputs


class MolBERTFeatureExtractor:
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
    
    def load_pretrained(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except ImportError:
            print("transformers library not installed. Using custom MolBERT implementation.")
            return False
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            return False
    
    def extract_features(self, smiles_list, batch_size=32):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        features = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            
            encoded = self.tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_features = outputs.pooler_output.cpu().numpy()
                features.append(batch_features)
        
        return np.concatenate(features, axis=0)


def create_molbert_model(vocab_size, d_model=256, num_heads=8, num_layers=4,
                         d_ff=1024, max_length=256, dropout=0.1, output_dim=None):
    return MolBERT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_length=max_length,
        dropout=dropout,
        output_dim=output_dim
    )


import numpy as np
