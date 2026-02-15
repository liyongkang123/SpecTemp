# from numba.core.types import none
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel,AutoTokenizer
from sentence_transformers import SentenceTransformer
# from FlagEmbedding import BGEM3FlagModel
from torch import Tensor
from sentence_transformers import SentenceTransformer

import os
# SentenceTransformer embeddings use mean pooling by default; only ReasonIR and Contriever use mean pooling here.

def load_model_hf(model_name):

    # When adding a new model here, update get_model_prompts_tasks as well, since some models need special prompts.
    
    # All embedding models except Contriever are configured with a maximum output length of 8192.

    model_kwargs = {
    "torch_dtype": torch.bfloat16,  # Use bfloat16 for better stability and performance.
    # "attn_implementation": "flash_attention_2",  # Use Flash Attention 2 for better efficiency.
    }

    if model_name=='reasonir':
        # encoder = SentenceTransformer("reasonir/ReasonIR-8B", trust_remote_code=True, model_kwargs=model_kwargs)
        # encoder.max_seq_length = 8192
        tokenizer = AutoTokenizer.from_pretrained("reasonir/ReasonIR-8B")
        model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype=torch.bfloat16, trust_remote_code=True,cache_dir=os.getenv('HF_HOME'))
        encoder = HFtoSF(model, tokenizer,  normalize=True , pooling='mask_prompt_mean', device='cuda')


    elif model_name == 'contriever':
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco") # this is the latest version
        model = AutoModel.from_pretrained("facebook/contriever-msmarco",torch_dtype=torch.bfloat16 , trust_remote_code=True) #torch_dtype=torch.bfloat16
        encoder = HFtoSF(model, tokenizer, normalize=False , pooling='mean', max_seq_length = 512, device='cuda') # use dot product

    elif model_name=='tas_b':
        tokenizer = AutoTokenizer.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
        model = AutoModel.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco", torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer, normalize=False , pooling='cls',max_seq_length = 512, device='cuda')
        
    elif model_name=='bge_reasoner':
        # last token
        # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reasoner-embed-qwen3-8b-0923")
        # model = AutoModel.from_pretrained("BAAI/bge-reasoner-embed-qwen3-8b-0923", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928") # this is the latest version
        model = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928", torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer, normalize=True , pooling='last', device='cuda')

    elif model_name=='diver':
        tokenizer = AutoTokenizer.from_pretrained('AQ-MedAI/Diver-Retriever-4B', padding_side='left')
        model = AutoModel.from_pretrained('AQ-MedAI/Diver-Retriever-4B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,normalize=True , pooling='last', device='cuda')

    elif model_name=='diver_1.7B':
        tokenizer = AutoTokenizer.from_pretrained('AQ-MedAI/Diver-Retriever-1.7B', padding_side='left')
        model = AutoModel.from_pretrained('AQ-MedAI/Diver-Retriever-1.7B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,normalize=True , pooling='last', device='cuda')
    elif model_name=='diver_0.6B':
        tokenizer = AutoTokenizer.from_pretrained('AQ-MedAI/Diver-Retriever-0.6B', padding_side='left')
        model = AutoModel.from_pretrained('AQ-MedAI/Diver-Retriever-0.6B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,normalize=True , pooling='last', device='cuda')

    elif model_name=='qwen3':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,   normalize=True ,pooling='last', device='cuda')
    elif model_name=='qwen3_0.6B':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,   normalize=True ,pooling='last', device='cuda')
    elif model_name=='qwen3_4B':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,   normalize=True ,pooling='last', device='cuda')

    elif model_name=='linq':
        tokenizer = AutoTokenizer.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral')
        model = AutoModel.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,  normalize=True , pooling='last', device='cuda')

    elif model_name == 'gte':
        tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True,torch_dtype=torch.bfloat16)
        model.config.use_cache = False
        encoder = HFtoSF(model, tokenizer, normalize=True , pooling='last', device='cuda')

    elif model_name=='bge_m3':
        model = AutoModel.from_pretrained('BAAI/bge-m3', trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        encoder = HFtoSF(model, tokenizer, normalize=True ,pooling='cls',  device='cuda')

    elif model_name=='embeddinggemma':
        model = SentenceTransformer("google/embeddinggemma-300m")
        tokenizer = None
        encoder = HFtoSF(model, tokenizer, normalize=True , device='cuda', model_name='embeddinggemma',max_seq_length=2048) # Uses cosine similarity; model_name must be set here.

    elif model_name=='jina_v4':
        model = AutoModel.from_pretrained('jinaai/jina-embeddings-v4', trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v4')
        encoder = HFtoSF(model, tokenizer, normalize=True , device='cuda', model_name='jina_v4') # Uses cosine similarity; model_name must be set here.

    elif model_name=='nomic_v2':
        model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v2-moe', trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v2-moe')
        encoder = HFtoSF(model, tokenizer, normalize=True , pooling='mean', device='cuda',max_seq_length=512) # Uses cosine similarity.

    elif model_name=="nv_embed_v2":
        model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2')
        encoder = HFtoSF(model, tokenizer, normalize=True , pooling='eos', device='cuda', model_name='nv_embed_v2')

    else:
        raise Exception('model_name error')

    return encoder,tokenizer

# Adapter to manually wrap an HF model into an SF-style interface by implementing encode().

class HFtoSF:
    def __init__(self, hf_model, hf_tokenizer,normalize= False, pooling='last', max_seq_length=8192 ,device='cuda', model_name= None):
        try:
            self.hf_model = hf_model.to(device)
            self.hf_model.eval()
        except: # for BGE M3, no attribute 'eval' and 'to'
            self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.max_seq_length = max_seq_length  # Tune this per model.
        self.pooling = pooling
        self.normalize = normalize

        self.model_name = model_name

    def _pooling(self, last_hidden_state, attention_mask, prompt=None):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['mask_prompt_mean']: # Follow ReasonIR pooling: mask prompt tokens before averaging.
            if prompt is None:
                prompt = self.prompt
            self.prompt_tokens = self.hf_tokenizer( prompt, padding=False, add_special_tokens=True, return_tensors='pt')
            attention_mask[:, :len(self.prompt_tokens['input_ids'][0])] = 0
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
 
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def tokenize_texts(self, texts):
        batch_dict = self.hf_tokenizer(texts, max_length=self.max_seq_length, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, pad_to_multiple_of=8)
        batch_dict = { k: v.to(self.device) for k, v in batch_dict.items() }
        return batch_dict

    def encode(self, texts, prompt, convert_to_numpy=False, show_progress_bar=False):
        # Input is a text list after batching.
        # Output is the corresponding embeddings.

        self.prompt = prompt  # Prompt is fixed for each encode call, so set it here.
        if isinstance(texts, str):
            texts = [texts]
        # Concatenate the prompt in front of each text.
        if self.model_name is None:

            if prompt is not None and prompt != '':
                texts = [prompt + text for text in texts]
            batch_inputs = self.tokenize_texts(texts)
            with torch.no_grad():
                outputs = self.hf_model(**batch_inputs)
                embeddings = self._pooling(outputs.last_hidden_state, batch_inputs['attention_mask'])
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            return embeddings


        elif self.model_name =='jina_v4': # Official usage is text-matching; validate carefully in real tests.
            # if prompt not in ('query', 'passage'):
            #     raise ValueError(f"prompt must be 'query' or 'passage', got '{prompt}'")
            embeddings = self.hf_model.encode_text(
                texts=texts,
                task="retrieval", # text-matching  retrieval
                prompt_name=prompt,
                batch_size =64,
                max_length=512,
            )
            embeddings = torch.stack(embeddings).cpu()
            if convert_to_numpy:
                embeddings = embeddings.numpy()
            return embeddings
        
        elif self.model_name=='embeddinggemma':
            if prompt=='query':
                embeddings = self.hf_model.encode_query(texts)
            elif prompt=='passage':
                embeddings = self.hf_model.encode_document(texts)
            # The embedding returned here is naturally a NumPy array.
            else:
                raise ValueError(f"prompt must be 'query' or 'passage', got '{prompt}'")
            return embeddings

        elif self.model_name =='nv_embed_v2':
            embeddings =self.hf_model.encode(texts, instruction=prompt, max_length=self.max_seq_length)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            return embeddings
