train_config = {
    "seed": 2023,
    "from_bert": False,
    "model": "RNN",
    "batch_size": 512,
    "sentence_length": 60,
    "learning_rate": 5e-4,
    "epoch": 100,
    "step_size": 5,
    "early_stop": True,
    "early_stop_steps": 10,
    "test_only": False,
    "ckpt": "lr_0.0005_bs_512_sota.ckpt"
}

base_config = {
    "num_embeddings": 59290, #! Do not change
    "embedding_dim": 50, #! Do not change
    "hidden_size": 64,
    "pretrained_embedding_path": "./Dataset/wiki_embedding.pth",
    "freeze_pretrained_embedding": False,
    "dropout_rate": 0.5,
    "init_config": {
       "method": "default"
    }
}

rnn_config = {
    "type": "LSTM",
    "num_layers": 2,
    "bidirectional": True
}