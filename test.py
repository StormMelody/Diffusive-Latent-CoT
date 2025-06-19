from transformers import GPT2Tokenizer

# 加载分词器，您可能需要根据项目中实际使用的模型名称调整，例如 "gpt2", "gpt2-medium" 等
# 如果您的项目有自定义的分词器路径，请替换 "gpt2" 为该路径
try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please ensure you have the correct tokenizer name or path.")
    exit()

token_id_to_check = 4211

# 使用 decode 方法将 token_id 转换为文本
# 注意：decode 通常用于解码一个序列，所以我们将单个 id 放入列表中
token_text = tokenizer.decode([token_id_to_check])

print(f"Token ID: {token_id_to_check}")
print(f"Corresponding Token Text: '{token_text}'")

# 或者，如果您想获取原始的 token 字符串（可能包含特殊前缀如 "Ġ" 表示空格）
# 可以使用 convert_ids_to_tokens
raw_token = tokenizer.convert_ids_to_tokens([token_id_to_check])
print(f"Raw Token (from convert_ids_to_tokens): {raw_token}")
