import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import collections
import numpy as np

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=False,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype="float16"
)

sentence_transformer = SentenceTransformer('BAAI/bge-large-en-v1.5')

llama_tokenizer = AutoTokenizer.from_pretrained("/model-weights/Llama-3.2-3B-Instruct/") # Llama-2-7b-hf/ Llama-3.2-3B-Instruct/
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"
llama_model = LlamaForCausalLM.from_pretrained(
    "/model-weights/Llama-3.2-3B-Instruct/", 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config, 
    # low_cpu_mem_usage=True, 
    # attn_implementation="flash_attention_2"
)
llama_model.eval()


bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

qwen_tokenizer = AutoTokenizer.from_pretrained("/model-weights/Qwen2.5-3B-Instruct/")
qwen_model = AutoModelForCausalLM.from_pretrained(
    "/model-weights/Qwen2.5-3B-Instruct/", 
    device_map="auto",
    quantization_config=bnb_config, 
)
qwen_model.eval()


CE_no_reduction = torch.nn.CrossEntropyLoss(reduction='none')


def get_sentence_transformer_embedding(sentences):
    embeddings = sentence_transformer.encode(sentences, convert_to_tensor=True)
    return embeddings.cpu()


def calculate_cosine_similarity(embedding1, embedding2):
    # Normalize the embeddings to unit vectors
    embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    similarity = torch.matmul(embedding1, embedding2.T)
    # print("[similarity]:", similarity.shape, similarity)
    # take the max value because we want to remove implications similar to image descriptions and large value will penalize final weight in information bottleneck theory
    max_similarity = torch.max(similarity, dim=-1).values.item()
    min_similarity = torch.min(similarity, dim=-1).values.item()
    return max_similarity, min_similarity


def calculate_entropy(target_token_ids):
    entropy_values = []
    for target_token_id in target_token_ids:
        # print(len(target_token_id), target_token_id[:10], collections.Counter(target_token_id))
        total_tokens = len(target_token_id)
        token_counts = collections.Counter(target_token_id)
        probabilities = np.array(list(token_counts.values())) / total_tokens
        # print("[probabilities]:", len(probabilities), probabilities)
        # Shannon entropy formula: H = -Σ p_i * log(p_i)
        entropy = -np.sum(probabilities * np.log(probabilities))
        # print("[entropy]:", entropy)
        entropy_values.append(entropy)
        
    return entropy_values


def calculate_length_penalized_cross_entropy(contexts, cross_entory_values):
    assert len(contexts) == len(cross_entory_values)
    
    if isinstance(cross_entory_values, list):
        cross_entory_values = torch.tensor(cross_entory_values)
    
    token_ids = llama_tokenizer(contexts)['input_ids']
    token_length = torch.tensor([len(x) for x in token_ids], dtype=torch.float)
    alpha = torch.mean(cross_entory_values) / torch.mean(token_length)
    mean_token_length = torch.mean(token_length)
    penalties = torch.abs(token_length - mean_token_length) * alpha
    penalized_cross_entory_values = cross_entory_values + penalties
    return penalized_cross_entory_values


def calculate_cross_entropy(contexts, targets, device, model):
    """
    Args:
        context (X): literal descriptive sentences
        target (Z): implications
    Returns:
        cross-entropy logits from GPT2
    """
    
    if model == "llama":
        tokenizer = llama_tokenizer
        model = llama_model
    elif model == "qwen2":
        tokenizer = qwen_tokenizer
        model = qwen_model
    else:
        raise Exception("Invalid model name!!")
    
    context_token_ids = tokenizer(contexts)['input_ids']
    target_token_ids = tokenizer(targets)['input_ids']

    assert len(context_token_ids) == 1
    assert len(target_token_ids) >= 1
    
    context_token_id = context_token_ids[0]
    # Convert target_token_ids to tensors and pad them for batch processing
    target_token_tensors = [torch.tensor(target_token_id, dtype=torch.long) for target_token_id in target_token_ids]
    target_token_padded = pad_sequence(target_token_tensors, batch_first=True) # [batch_size, padded_size]
    
    # Create input batch by concatenating the context with each target
    batch_size = target_token_padded.shape[0]
    context_tensor = torch.tensor(context_token_id, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    input_tensor = torch.cat([context_tensor, target_token_padded], dim=-1)

    # Create target mask: 1 for target tokens, 0 for context tokens
    target_mask = torch.zeros_like(input_tensor, dtype=torch.long)
    target_lengths = [len(t) for t in target_token_ids]
    for i, target_len in enumerate(target_lengths):
        target_mask[i, len(context_token_id):len(context_token_id) + target_len] = 1

    # Perform inference on batch
    with torch.no_grad():
        logits = model(input_tensor.to(device))['logits']  # [batch_size, seq_len, vocab_size]
        logits = logits[:, :-1].cpu().contiguous()  # Shifted logits

    # Prepare target tensor for loss calculation
    target_tensor = input_tensor[:, 1:].cpu().contiguous()  # Target is the shifted input
    target_mask = target_mask[:, 1:]
    
    # Cross-entropy loss calculation
    ce_matrix = CE_no_reduction(logits.view(-1, logits.shape[-1]), target_tensor.view(-1))  # [batch_size * (seq_len-1)]
    ce_matrix = ce_matrix.view(logits.shape[:-1])

    # Mask cross-entropy results for the target tokens
    non_zero_count = target_mask.sum(dim=-1)
    target_ce_list = (ce_matrix * target_mask).sum(dim=-1) / non_zero_count
    
    # print("target_ce_list:", target_ce_list.cpu().tolist())
    # print("target_token_length:", target_token_length)
    
    # entropy_values = calculate_entropy(target_token_ids)
    # penalized_target_ce_list = target_ce_list 
    # normalized_target_ce_list = target_ce_list / torch.log(torch.tensor(target_token_length)+1)
    # print("penalties:", penalties)
    # print("penalized_target_ce_list:", penalized_target_ce_list)
    return target_ce_list.cpu().tolist() # .cpu() to prevent cuda memory issue


def calculate_batched_cross_entropy(batch_size, input_prompt, targets, device):
    """
        contexts: input_prompt
        targets: candidate_responses
    """
    all_cross_entropy_values = []
    for i in range(0, len(targets), batch_size):
        batched_candidate_responses = targets[i:min(i+batch_size, len(targets))]
        if not batched_candidate_responses:
            break
        cross_entropy_values = calculate_cross_entropy([input_prompt], batched_candidate_responses, device, "qwen2")
        all_cross_entropy_values.extend(cross_entropy_values)
    
    if len(targets) > 2: # penalizing is meaningful only when there are more than 2 sentences, otherwise CE score remain the same (length=1) or increased with the same gap (length=2)
        all_cross_entropy_values = calculate_length_penalized_cross_entropy(targets, all_cross_entropy_values).tolist()
        # print("penalized_cross_entropy_values:", all_cross_entropy_values)
    # print("all_cross_entropy_values:", all_cross_entropy_values)
    return all_cross_entropy_values
