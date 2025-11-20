# BERT from Scratch for IMDB Classification

This directory contains a complete implementation of BERT from scratch (using PyTorch) and a training pipeline for the IMDB sentiment classification task.

## Files

- `config.py`: Configuration class for BERT hyperparameters.
- `model.py`: Implementation of the BERT model architecture (Embeddings, SelfAttention, Encoder, Pooler).
- `dataset.py`: Data loading and tokenization using the IMDB dataset.
- `train.py`: Training script with evaluation.
- `requirements.txt`: Python dependencies.

## Usage

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the model:**

    Run the training script. You can adjust hyperparameters via command line arguments.

    ```bash
    # Run as a module to ensure relative imports work
    python -m bert_from_scratch.train --epochs 3 --batch_size 8
    ```

    **Note on running:** Since the files use relative imports (e.g., `from .config import BertConfig`), you should run the script from the parent directory (the one containing `bert_from_scratch`).

    Example from the workspace root:
    ```bash
    python -m bert_from_scratch.train
    ```

    **Arguments:**
    - `--batch_size`: Batch size (default: 8)
    - `--lr`: Learning rate (default: 2e-5)
    - `--epochs`: Number of training epochs (default: 3)
    - `--max_length`: Maximum sequence length (default: 128, increase to 512 for full length if you have enough VRAM)
    - `--num_layers`: Number of transformer layers (default: 12)
    - `--hidden_size`: Hidden size (default: 768)
    - `--num_heads`: Number of attention heads (default: 12)

## Implementation Details

- The model architecture matches `bert-base-uncased`.
- It uses a standard `BertTokenizer` from the `transformers` library for tokenization, but the model itself is implemented purely in PyTorch in `model.py`.
- The IMDB dataset is downloaded automatically via the `datasets` library.

