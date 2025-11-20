import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer

class IMDBDataset(Dataset):
    def __init__(self, split="train", max_length=512, tokenizer_name="bert-base-uncased"):
        self.dataset = load_dataset("imdb", split=split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def get_dataloader(split="train", batch_size=8, max_length=512):
    dataset = IMDBDataset(split=split, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

