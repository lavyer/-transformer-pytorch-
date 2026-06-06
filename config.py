from pathlib import Path

def get_config():
    return {
        # ----- Training Config -----
        "batch_size": 64,             # Batch size
        "num_epochs": 15,            # Number of epochs
        "lr": 1.0,                   # Base LR (actual lr comes from warmup formula)
        "seq_len": 50,               # Max sequence length
        "max_train_samples": None,   # Use all available data (~30K pairs)
        # ----- Model Config -----
        "d_model": 256,              # Embedding dimension
        "n_heads": 8,                # Number of attention heads
        "n_layers": 3,               # Number of encoder/decoder layers
        "d_ff": 1024,                # Feed-forward network dimension
        "dropout": 0.1,              # Dropout rate
        "english_vocab_size": 10000, # English vocabulary size cap
        # ----- Data Config -----
        "datasource": 'cmn-eng',     # Dataset name
        "data_dir": 'cmn-eng (1)',   # Data directory
        "lang_src": "zh",            # Source language: Chinese
        "lang_tgt": "en",            # Target language: English
        # ----- File Config -----
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,             # Train from scratch
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_zh_en_v2"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
