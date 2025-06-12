import pickle
import torch
import json

def check_pkl_file(pkl_path, original_json_path=None):
    """
    检查pkl文件的内容和结构
    """
    print(f"检查文件: {pkl_path}")
    
    try:
        # 加载pkl文件
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ 成功加载pkl文件")
        print(f"✓ 总样本数: {len(data)}")
        
        if len(data) == 0:
            print("❌ 错误: pkl文件为空")
            return False
        
        # 检查第一个样本的结构
        first_sample = data[0]
        print(f"\n第一个样本的keys: {list(first_sample.keys())}")
        
        # 检查必需的字段
        required_keys = ['idx', 'question_embedding', 'answer_token_ids', 'steps_embeddings']
        missing_keys = [key for key in required_keys if key not in first_sample]
        if missing_keys:
            print(f"❌ 错误: 缺少必需字段: {missing_keys}")
            return False
        else:
            print("✓ 所有必需字段都存在")
        
        # 检查数据类型和形状
        print("\n数据类型和形状检查:")
        print(f"  idx: {first_sample['idx']} (type: {type(first_sample['idx'])})")
        print(f"  question_embedding shape: {first_sample['question_embedding'].shape}")
        print(f"  question_embedding dtype: {first_sample['question_embedding'].dtype}")
        print(f"  answer_token_ids shape: {first_sample['answer_token_ids'].shape}")
        print(f"  answer_token_ids dtype: {first_sample['answer_token_ids'].dtype}")
        print(f"  steps_embeddings shape: {first_sample['steps_embeddings'].shape}")
        print(f"  steps_embeddings dtype: {first_sample['steps_embeddings'].dtype}")
        
        # 检查embedding维度是否合理
        if first_sample['question_embedding'].shape[-1] != 1024:  # DialoGPT-medium的hidden size
            print(f"⚠️  警告: question_embedding维度不是1024: {first_sample['question_embedding'].shape[-1]}")
        
        if first_sample['steps_embeddings'].shape[-1] != 1024:
            print(f"⚠️  警告: steps_embeddings维度不是1024: {first_sample['steps_embeddings'].shape[-1]}")
        
        # 检查是否有NaN或无穷值
        if torch.isnan(first_sample['question_embedding']).any():
            print("❌ 错误: question_embedding包含NaN值")
            return False
        
        if torch.isnan(first_sample['steps_embeddings']).any():
            print("❌ 错误: steps_embeddings包含NaN值")
            return False
        
        if torch.isinf(first_sample['question_embedding']).any():
            print("❌ 错误: question_embedding包含无穷值")
            return False
        
        if torch.isinf(first_sample['steps_embeddings']).any():
            print("❌ 错误: steps_embeddings包含无穷值")
            return False
        
        print("✓ 没有发现NaN或无穷值")
        
        # 检查几个样本的一致性
        print("\n检查样本一致性:")
        for i in range(min(5, len(data))):
            sample = data[i]
            if sample['question_embedding'].shape[-1] != first_sample['question_embedding'].shape[-1]:
                print(f"❌ 错误: 样本{i}的question_embedding维度不一致")
                return False
            if sample['steps_embeddings'].shape[-1] != first_sample['steps_embeddings'].shape[-1]:
                print(f"❌ 错误: 样本{i}的steps_embeddings维度不一致")
                return False
        
        print("✓ 前5个样本的embedding维度一致")
        
        # 如果提供了原始JSON文件，检查数量是否匹配
        if original_json_path:
            try:
                with open(original_json_path, 'r') as f:
                    original_data = json.load(f)
                
                expected_count = len(original_data)
                actual_count = len(data)
                
                if actual_count == expected_count:
                    print(f"✓ 样本数量匹配原始数据: {actual_count}/{expected_count}")
                else:
                    print(f"⚠️  样本数量不匹配: pkl有{actual_count}个，原始JSON有{expected_count}个")
                    print(f"   可能是因为跳过了一些无效样本")
            except Exception as e:
                print(f"⚠️  无法检查原始JSON文件: {e}")
        
        print("\n✅ pkl文件检查完成，没有发现严重问题")
        return True
        
    except Exception as e:
        print(f"❌ 错误: 无法加载pkl文件: {e}")
        return False

if __name__ == "__main__":
    # 检查debug版本的pkl文件
    debug_pkl = "/data2/xxw_data/projects/LLM/coconut/data/gsm_train_embeddings_debug.pkl"
    debug_json = "/data2/xxw_data/projects/LLM/coconut/data/gsm_train_debug.json"
    
    print("=" * 60)
    print("检查Debug版本的pkl文件")
    print("=" * 60)
    check_pkl_file(debug_pkl, debug_json)
    
    # 如果完整版本存在，也检查一下
    full_pkl = "/data2/xxw_data/projects/LLM/coconut/data/gsm_train_embeddings.pkl"
    full_json = "/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json"
    
    import os
    if os.path.exists(full_pkl):
        print("\n" + "=" * 60)
        print("检查完整版本的pkl文件")
        print("=" * 60)
        check_pkl_file(full_pkl, full_json)
    else:
        print(f"\n完整版本的pkl文件不存在: {full_pkl}")