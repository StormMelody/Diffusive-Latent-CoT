from transformers import GPT2Tokenizer

# 加载分词器，您可能需要根据项目中实际使用的模型名称调整，例如 "gpt2", "gpt2-medium" 等
# 如果您的项目有自定义的分词器路径，请替换 "gpt2" 为该路径
try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please ensure you have the correct tokenizer name or path.")
    exit()
tokenizer.add_tokens("<BOD>")
tokenizer.add_tokens("<EOD>")  
start_id = tokenizer.convert_tokens_to_ids("<BOD>") # 将这个标识转化为id，这个id后续会添加到embedding查找表中。id可以通过索引寻找对应的embedding。
end_id = tokenizer.convert_tokens_to_ids("<EOD>")
token_ids_to_check = [   32, 15900,  5318,   468,   257,  1728,  1271,   286, 15900,  7317,
           13, 10923,  1443,   389,  4642,   379,   262,  2494,   286,   838,
          583,  1227,   290, 15900,  4656,   379,   262,  2494,   286,   362,
          583,  1227,    13,  2293,   352,   614,    11,   612,   389,   939,
        15900,   319,   262,  5318,    13,  1374,   867, 15900,   547,   612,
          319,   262,  5318,  6198,    30,   198, 50257, 16791,   940,    12,
           17,    28,    23,  4211,   198, 16791,    23,     9,  1065,    28,
         4846,  4211,   198, 16791,  2167,    12,  4846,    28, 13464,  4211,
          198, 50258, 21017, 14436, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]
# 使用 decode 方法将 token_id 序列转换为文本
token_text = tokenizer.decode(token_ids_to_check)

print(f"Token IDs: {token_ids_to_check}")
print(f"Corresponding Token Text: '{token_text}'")

# 或者，如果您想获取原始的 token 字符串（可能包含特殊前缀如 "Ġ" 表示空格）
# 可以使用 convert_ids_to_tokens
raw_tokens = tokenizer.convert_ids_to_tokens(token_ids_to_check)
print(f"Raw Tokens (from convert_ids_to_tokens): {raw_tokens}")
