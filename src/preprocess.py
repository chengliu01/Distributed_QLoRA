import torch
from transformers import AutoTokenizer
from torch.utils.data import random_split, Subset
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from jinja2 import Template
import numpy as np
from utils import print_rank0

class InstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sources = self.data[index]
        if isinstance(index, int):
            sources = [sources]
        data_dict = preprocess([e['conversations'] for e in sources], self.tokenizer, self.max_seq_len)
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: AutoTokenizer
    IGNORE_INDEX = -100

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def prepare_data_splits():
    """
        You could implement your own method!!!!
    """
    try:
        print_rank0("Loading Lawyer-Instruct ...")
        dataset = load_dataset("Alignment-Lab-AI/Lawyer-Instruct")
        train_data = dataset['train']
        print_rank0(f"Completed loading: {len(train_data)} samples")
        np.random.seed(42)
        
        test_size = min(100, len(train_data))
        all_indices = np.random.permutation(len(train_data))
        test_indices = all_indices[:test_size]
        remaining_indices = all_indices[test_size:]
        
        train_val_size = len(remaining_indices)
        train_size = int(train_val_size * 0.9)
        val_size = train_val_size - train_size
        
        train_indices = remaining_indices[:train_size]
        val_indices = remaining_indices[train_size:train_size+val_size]
        
        train_subset = Subset(train_data, train_indices)
        val_subset = Subset(train_data, val_indices)
        test_subset = Subset(train_data, test_indices)
        
        print_rank0(f"Data has been split：Train size {len(train_subset)} , Val size {len(val_subset)} , Test size {len(test_subset)}")
        
        return train_subset, val_subset, test_subset
    
    except Exception as e:
        print_rank0(f"Data Wrong: {e}")
        return None, None, None


def prepare_test_data():
    train_subset, val_subset, test_subset = prepare_data_splits()
    test_data = []
    for i in range(len(test_subset)):
        
        idx = test_subset.indices[i]
        item = test_subset.dataset[idx:idx+1]
        test_data.append({
            'instruction': item['instruction'][0],
            'output': item['output'][0]
        })
    test_dataset = Dataset.from_list(test_data)
    print_rank0(f"Testset size: {len(test_dataset)}")
    return test_dataset


def prepare_data():
    try:
        train_subset, val_subset, _ = prepare_data_splits()
        if train_subset is None or val_subset is None:
            return None, None

        def convert_format(sample):
            if isinstance(sample, dict):
                return {
                    "conversations": [
                        {"from": "human", "value": sample['instruction'][0]},
                        {"from": "gpt", "value": sample['output'][0]}
                    ]
                }
            index = sample
            if isinstance(index, np.integer):
                index = int(index)
            original_sample = train_subset.dataset[index] if hasattr(sample, 'dataset') else sample
            return {
                "conversations": [
                    {"from": "human", "value": original_sample['instruction'][0]},
                    {"from": "gpt", "value": original_sample['output'][0]}
                ]
            }
        
        mapped_train = [convert_format(train_subset[i:i+1]) for i in range(len(train_subset))]
        mapped_val = [convert_format(val_subset[i:i+1]) for i in range(len(val_subset))]
        
        print_rank0(f"Trainset size: {len(mapped_train)}, Val size: {len(mapped_val)}")
        return mapped_train, mapped_val
    
    except Exception as e:
        print_rank0(f"Data Wrong: {e}")
        return None, None


def preprocess(sources, tokenizer, max_seq_len=1024):
    template = Template(tokenizer.chat_template)
    max_seq_len = min(max_seq_len, tokenizer.model_max_length)
    messages = []
    
    legal_prompt = "You are a professional legal consultant. Please provide accurate and professional legal advice based on the following questions, ensuring the precision of legal terminology and the completeness of responses."
    
    for source in sources:
        if source[0]["from"] != "human":
            source = source[1:]

        for j in range(0, len(source), 2):
            if j + 1 >= len(source):
                continue
            q = f"{legal_prompt}\n\n{source[j]['value']}"
            a = source[j + 1]["value"]
            assert q is not None and a is not None, f'q:{q} a:{a}'
            
            input = template.render(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                bos_token=tokenizer.bos_token, add_generation_prompt=False
            )
            input_ids = tokenizer.encode(input, add_special_tokens=False)

            query = template.render(
                messages=[{"role": "user", "content": q}],
                bos_token=tokenizer.bos_token,
                add_generation_prompt=True
            )
            query_ids = tokenizer.encode(query, add_special_tokens=False)

            labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
            assert len(labels) == len(input_ids)
            if len(input_ids) == 0:
                continue
            messages.append({"input_ids": input_ids[-max_seq_len:], "labels": labels[-max_seq_len:]})

    input_ids = [item["input_ids"] for item in messages]
    labels = [item["labels"] for item in messages]

    max_len = max(len(x) for x in input_ids)
    max_len = min(max_len, max_seq_len)
    input_ids = [item[:max_len] + [tokenizer.eos_token_id] * (max_len - len(item)) for item in input_ids]
    labels = [item[:max_len] + [-100] * (max_len - len(item)) for item in labels]

    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def __test__():
    test_dataset = prepare_test_data()
    data, _ = prepare_data()
    print(data[0])
    
    
if __name__ == "__main__":
    __test__()