import json
import torch
import pickle
import os
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

class TextDataset(Dataset):
    """用于批处理的文本数据集"""
    def __init__(self, texts: List[str], sample_indices: List[Tuple[int, str, int]]):
        self.texts = texts
        self.sample_indices = sample_indices  # (sample_idx, type, step_idx)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.sample_indices[idx]

def collate_fn(batch):
    """自定义collate函数"""
    texts, indices = zip(*batch)
    return list(texts), list(indices)

def preprocess_embeddings_optimized(data_path, output_path, model_name="openai-community/gpt2", 
                                   device="cuda", batch_size=32, num_workers=4, use_fp16=False, data_input=None):
    """
    优化版本的预处理函数，使用批处理和并行处理加速
    
    Args:
        data_path: 输入数据路径 (如果提供了data_input，则此项可选)
        output_path: 输出路径
        model_name: 模型名称
        device: 设备
        batch_size: 批处理大小
        num_workers: 数据加载器工作进程数
        use_fp16: 是否使用半精度
        data_input: 预加载的数据 (可选)
    """
    if data_input is not None:
        data = data_input
        print(f"Using preloaded data with {len(data)} samples...")
    elif data_path is not None:
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Either data_path or data_input must be provided.")
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    embedding_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    embedding_model.eval()
    
    # 使用半精度优化
    if use_fp16 and device == "cuda":
        embedding_model = embedding_model.half()
        print("Using FP16 for faster inference")
    
    # 尝试使用torch.compile加速（PyTorch 2.0+）
    try:
        embedding_model = torch.compile(embedding_model)
        print("Using torch.compile for acceleration")
    except:
        print("torch.compile not available, using standard model")
    
    # 第一步：收集所有需要处理的文本
    print("Collecting texts for batch processing...")
    all_texts = []
    text_to_sample_map = []  # (sample_idx, type, step_idx)
    valid_samples = []
    
    for idx, sample in enumerate(data):
        question = sample['question']
        answer = sample['answer']
        original_steps = ['<BOD>'] + sample['steps'] + ['<EOD>']
        
        # 跳过空steps的样本
        if len(sample['steps']) == 0:
            print(f"Warning: Skipping sample {idx} with empty steps")
            continue
        
        # 记录有效样本
        valid_samples.append((idx, sample))
        
        # 添加question
        all_texts.append(question)
        text_to_sample_map.append((len(valid_samples)-1, 'question', 0))
        
        # 添加所有steps
        for step_idx, step in enumerate(original_steps):
            if step:  # 跳过空字符串
                all_texts.append(step)
                text_to_sample_map.append((len(valid_samples)-1, 'step', step_idx))
    
    print(f"Total texts to process: {len(all_texts)}")
    print(f"Valid samples: {len(valid_samples)}")
    
    # 第二步：批处理计算embeddings
    print("Computing embeddings in batches...")
    dataset = TextDataset(all_texts, text_to_sample_map)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )
    
    all_embeddings = []
    all_indices = []
    
    with torch.no_grad():
        for batch_texts, batch_indices in tqdm(dataloader, desc="Processing batches"):
            # Tokenize batch
            batch_tokens = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(device)
            
            # 计算embeddings
            outputs = embedding_model(**batch_tokens, output_hidden_states=True)
            batch_embeddings = outputs.hidden_states[-1].mean(dim=1).cpu()
            
            # 如果使用FP16，转换回FP32保存
            if use_fp16:
                batch_embeddings = batch_embeddings.float()
            
            all_embeddings.extend(batch_embeddings)
            all_indices.extend(batch_indices)
    
    print(f"Computed {len(all_embeddings)} embeddings")
    
    # 第三步：重组数据
    print("Reorganizing data...")
    processed_data = []
    
    # 为每个有效样本创建embedding字典
    sample_embeddings = {}
    for embedding, (sample_idx, text_type, step_idx) in zip(all_embeddings, all_indices):
        if sample_idx not in sample_embeddings:
            sample_embeddings[sample_idx] = {'question': None, 'steps': {}}
        
        if text_type == 'question':
            sample_embeddings[sample_idx]['question'] = embedding.unsqueeze(0)  # [1, 768]
        else:  # step
            sample_embeddings[sample_idx]['steps'][step_idx] = embedding  # [768]
    
    # 构建最终的processed_data
    for sample_idx, (original_idx, sample) in enumerate(valid_samples):
        try:
            question = sample['question']
            answer = sample['answer']
            
            # 获取question embedding
            question_embedding = sample_embeddings[sample_idx]['question']
            
            # 获取steps embeddings
            steps_dict = sample_embeddings[sample_idx]['steps']
            if len(steps_dict) == 0:
                print(f"Warning: Skipping sample {original_idx} with no valid steps")
                continue
            
            # 按step_idx排序并堆叠
            sorted_steps = sorted(steps_dict.items())
            steps_embeddings = torch.stack([embedding for _, embedding in sorted_steps], dim=0)
            
            # 计算answer token ids
            answer_tokens = tokenizer(
                answer + tokenizer.eos_token, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            answer_token_ids = answer_tokens.input_ids
            
            # 保存处理后的数据
            processed_sample = {
                'idx': original_idx,
                'question_embedding': question_embedding,
                'answer_token_ids': answer_token_ids,
                'steps_embeddings': steps_embeddings,
                'original_question': question,
                'original_answer': answer,
                'original_steps': sample['steps']
            }
            
            processed_data.append(processed_sample)
            
        except Exception as e:
            print(f"Error processing sample {original_idx}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_data)} samples")
    print(f"Saving to {output_path}...")
    
    # 保存处理后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Preprocessing complete! Saved {len(processed_data)} samples to {output_path}")
    return len(processed_data)

def preprocess_embeddings_chunked(data_path, output_path, model_name="openai-community/gpt2", 
                                 device="cuda", batch_size=32, num_workers=4, use_fp16=False,
                                 chunk_size=1000):
    """
    分块处理版本，适用于超大数据集，避免内存溢出
    """
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    print(f"Processing {len(data)} samples in {total_chunks} chunks of size {chunk_size}")
    
    all_processed_data = []
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(data))
        chunk_data = data[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks} (samples {start_idx}-{end_idx-1})")
        
        # 创建临时文件路径
        temp_output = output_path.replace('.pkl', f'_temp_chunk_{chunk_idx}.pkl')
        
        # 处理当前chunk
        chunk_processed = preprocess_embeddings_optimized(
            data_path=None,  # data_path is not used when data_input is provided
            output_path=temp_output,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            use_fp16=use_fp16,
            data_input=chunk_data  # 直接传入数据
        )
        
        # 加载处理后的数据
        with open(temp_output, 'rb') as f:
            chunk_result = pickle.load(f)
        all_processed_data.extend(chunk_result)
        
        # 删除临时文件
        os.remove(temp_output)
    
    # 保存最终结果
    print(f"\nSaving final result with {len(all_processed_data)} samples to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(all_processed_data, f)
    
    return len(all_processed_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized preprocessing for GSM8K dataset")
    parser.add_argument('--data_path', type=str, 
                       default="/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json", 
                       help='Path to input JSON file')
    parser.add_argument('--output_path', type=str, 
                       default="/data2/xxw_data/projects/LLM/coconut/data/gsm_train_embeddings_optimized.pkl", 
                       help='Path to output pickle file')
    parser.add_argument('--model_name', type=str, default="openai-community/gpt2", 
                       help='Model name for embeddings')
    parser.add_argument('--device', type=str, default="cuda", 
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    parser.add_argument('--use_fp16', action='store_true', 
                       help='Use half precision for faster inference')
    parser.add_argument('--chunked', action='store_true', 
                       help='Use chunked processing for large datasets')
    parser.add_argument('--chunk_size', type=int, default=1000, 
                       help='Chunk size for chunked processing')
    parser.add_argument('--debug', action='store_true', 
                       help='Process only first 100 samples for debugging')
    
    args = parser.parse_args()
    
    # Debug模式下只处理前1000个样本
    if args.debug:
        print("Debug mode: processing only first 1000 samples")
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        data = data[:1000]
        debug_data_path = args.data_path.replace('.json', '_debug.json')
        with open(debug_data_path, 'w') as f:
            json.dump(data, f)
        args.data_path = debug_data_path
        args.output_path = args.output_path.replace('.pkl', '_debug.pkl')
    
    # 选择处理方式
    if args.chunked:
        preprocess_embeddings_chunked(
            args.data_path, args.output_path, args.model_name, args.device,
            args.batch_size, args.num_workers, args.use_fp16, args.chunk_size
        )
    else:
        preprocess_embeddings_optimized(
            args.data_path, args.output_path, args.model_name, args.device,
            args.batch_size, args.num_workers, args.use_fp16
        )