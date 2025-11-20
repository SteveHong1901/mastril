import sys
import os
from pathlib import Path

# Add the bert_from_scratch directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

import torch
from transformers import BertTokenizer
import argparse
from config import ModelConfig
from model import TransformerClassifier


def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model with same config as training
    config = ModelConfig(
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=4
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
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def predict(model, tokenizer, text, device, max_length=128):
    """Predict sentiment for a single text input."""
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    # Get probabilities and prediction
    probs = torch.softmax(logits, dim=-1)
    prediction = torch.argmax(logits, dim=-1).item()
    confidence = probs[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence, probs[0].tolist()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    print("\n" + "="*60)
    print("IMDB Sentiment Classifier - Inference Mode")
    print("="*60 + "\n")
    
    if args.text:
        # Single text inference
        text = args.text
        print(f"Input text: {text}\n")
        sentiment, confidence, probs = predict(model, tokenizer, text, device, args.max_length)
        print(f"Prediction: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: [Negative: {probs[0]:.4f}, Positive: {probs[1]:.4f}]")
    
    elif args.interactive:
        # Interactive mode
        print("Interactive mode - Enter text to classify (or 'quit' to exit)\n")
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            if not text:
                print("Please enter some text.")
                continue
            
            sentiment, confidence, probs = predict(model, tokenizer, text, device, args.max_length)
            print(f"\nPrediction: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Probabilities: [Negative: {probs[0]:.4f}, Positive: {probs[1]:.4f}]")
    
    else:
        # Demo mode with example texts
        example_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible film. Waste of time and money. Very disappointed.",
            "Not bad, but not great either. Just okay.",
            "Best movie I've seen this year! Highly recommended!",
            "Boring and predictable. The acting was awful.",
        ]
        
        print("Demo mode - Running inference on example texts:\n")
        for i, text in enumerate(example_texts, 1):
            print(f"\n{i}. Text: {text}")
            sentiment, confidence, probs = predict(model, tokenizer, text, device, args.max_length)
            print(f"   Prediction: {sentiment}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Probabilities: [Negative: {probs[0]:.4f}, Positive: {probs[1]:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Sentiment Classifier Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to classify (if not provided, runs in demo mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    main(args)

