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
            # padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0), # (1, L) -> (L,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def collate_fn(batch):
    """Batch-level padding (dynamic padding)"""
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=0
    ) # pad sequence stack the things for us

    attention_mask = pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack([item['labels'] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }



def get_dataloader(split="train", batch_size=8, max_length=512):
    dataset = IMDBDataset(split=split, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_fn)

