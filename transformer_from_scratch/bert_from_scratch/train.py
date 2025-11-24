import sys
import os
from pathlib import Path

# Add the bert_from_scratch directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

import torch
from torch.optim import AdamW
from tqdm import tqdm
import argparse
from config import ModelConfig
from model import TransformerClassifier
from dataset import get_dataloader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    config = ModelConfig(
        num_labels=2,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        num_hidden_layers=args.num_layers, 
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads
    )
    
    model = TransformerClassifier(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        pad_token_id=config.pad_token_id,
        max_position_embeddings=config.max_position_embeddings,
        type_vocab_size=config.type_vocab_size,
        layer_norm_eps=config.layer_norm_eps,
        hidden_dropout_prob=config.hidden_dropout_prob,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        num_labels=config.num_labels
    )
    model.to(device)

    # 2. DataLoaders
    print("Loading data...")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size, max_length=args.max_length)
    test_loader = get_dataloader(split="test", batch_size=args.batch_size, max_length=args.max_length)

    # 3. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")
        
        # 5. Evaluation
        evaluate(model, test_loader, device)
        
        # Save checkpoint
        checkpoint_dir = current_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    
    args = parser.parse_args()
    train(args)
