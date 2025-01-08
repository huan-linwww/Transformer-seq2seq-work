from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from conf import *

# 加载支持中英文的分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
dataset = load_dataset("parquet", data_files={"train": ["data/train-00000-of-00013.parquet", "data/train-00001-of-00013.parquet", "data/train-00002-of-00013.parquet"], "test": "data/test-00000-of-00001.parquet", "validation": "data/validation-00000-of-00001.parquet"})

#批量分词
def process_data(examples):
    #print(examples)
    data = examples["translation"]
    input = tokenizer([item['zh'] for item in data], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")#tokenizer() 支持批量处理能批量处理数据
    #print(input) #返回一个字典，包含input_ids, token_type_ids, attention_mask
    label = tokenizer([item['en'] for item in data], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return {"input": input['input_ids'], "label": label['input_ids'], "attention_mask": input['attention_mask']}

    

#  对数据集进行批量分词
encoded_dataset = dataset.map(process_data, batched=True)
train_dataset = encoded_dataset["train"]
test_dataset = encoded_dataset["test"]
val_dataset = encoded_dataset["validation"]

train_dataloader = DataLoader(encoded_dataset["train"], batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(encoded_dataset["validation"], batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(encoded_dataset["test"], batch_size=batch_size, shuffle=False)

pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
vocab_size = tokenizer.vocab_size
sos_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)#用cls作为sos
if __name__ == '__main__':
    print(dataset)
    # 访问数据示例
    print(dataset['train'][0])  # 查看第一条训练数据
    
     # 查看每个特殊符号的 ID
    # pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    # sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    # unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    # mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # print(f"Pad ID: {pad_id}")
    # print(f"CLS ID: {cls_id}")
    # print(f"SEP ID: {sep_id}")
    # print(f"UNK ID: {unk_id}")
    # print(f"MASK ID: {mask_id}")
    print(encoded_dataset["train"][0])
    

