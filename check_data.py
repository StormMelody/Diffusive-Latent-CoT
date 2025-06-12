import json

# 加载数据
with open('/data2/xxw_data/projects/LLM/coconut/data/gsm_train.json', 'r') as f:
    data = json.load(f)

print('Total samples:', len(data))
print('Sample structure:')
print(json.dumps(data[0], indent=2, ensure_ascii=False))
print('\nFirst 3 samples:')
for i in range(min(3, len(data))):
    print(f'Sample {i}:')
    print(f'  Question: {data[i]["question"][:100]}...')
    print(f'  Steps: {data[i]["steps"][:3]}...')
    print(f'  Answer: {data[i]["answer"]}')
    print()