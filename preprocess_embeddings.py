import json
import torch
import pickle
import os
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import argparse

def preprocess_embeddings(data_path, output_path, model_name="openai-community/gpt2", device="cuda"):
    """
    预处理数据集，计算所有question和steps的embeddings并保存
    """
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    embedding_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    embedding_model.eval()
    
    processed_data = []
    
    print(f"Processing {len(data)} samples...")
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data)):
            try:
                question = sample['question']
                answer = sample['answer']
                original_steps = ['<BOD>'] + sample['steps'] + ['<EOD>']
                
                # 跳过空steps的样本
                if len(sample['steps']) == 0:
                    print(f"Warning: Skipping sample {idx} with empty steps")
                    continue
                
                # 计算question embedding
                question_tokens = tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(device)
                question_output = embedding_model(**question_tokens, output_hidden_states=True)
                question_embedding = question_output.hidden_states[-1].mean(dim=1).cpu()  # [1, 768]
                
                # 计算answer token ids
                answer_tokens = tokenizer(answer + tokenizer.eos_token, padding=True, truncation=True, return_tensors='pt')
                answer_token_ids = answer_tokens.input_ids
                
                # 计算steps embeddings
                steps_embeddings = []
                for step in original_steps:
                    if not step:  # 跳过空字符串
                        continue
                    step_tokens = tokenizer(step, padding=True, truncation=True, return_tensors='pt').to(device)
                    step_output = embedding_model(**step_tokens, output_hidden_states=True)
                    step_embedding = step_output.hidden_states[-1].squeeze(0).mean(dim=0).cpu()  # [768]
                    steps_embeddings.append(step_embedding)
                
                if len(steps_embeddings) == 0:
                    print(f"Warning: Skipping sample {idx} with no valid steps")
                    continue
                
                steps_embeddings = torch.stack(steps_embeddings, dim=0)  # [num_steps, 768]
                
                # 保存处理后的数据
                processed_sample = {
                    'idx': idx,
                    'question_embedding': question_embedding,
                    'answer_token_ids': answer_token_ids,
                    'steps_embeddings': steps_embeddings,
                    'original_question': question,
                    'original_answer': answer,
                    'original_steps': sample['steps']
                }
                
                processed_data.append(processed_sample)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    print(f"Successfully processed {len(processed_data)} samples")
    print(f"Saving to {output_path}...")
    
    # 保存处理后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Preprocessing complete! Saved {len(processed_data)} samples to {output_path}")
    return len(processed_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess embeddings for GSM8K dataset")
    parser.add_argument('--data_path', type=str, default="/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json", 
                       help='Path to input JSON file')
    parser.add_argument('--output_path', type=str, default="/data2/xxw_data/projects/LLM/coconut/data/gsm_train_embeddings.pkl", 
                       help='Path to output pickle file')
    parser.add_argument('--model_name', type=str, default="openai-community/gpt2", 
                       help='Model name for embeddings')
    parser.add_argument('--device', type=str, default="cuda", 
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', 
                       help='Process only first 100 samples for debugging')
    
    args = parser.parse_args()
    
    # Debug模式下只处理前100个样本
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
    
    preprocess_embeddings(args.data_path, args.output_path, args.model_name, args.device)