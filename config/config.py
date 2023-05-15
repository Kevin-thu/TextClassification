"""
Set the config here.
"""
train_config = {
    "seed": 2023,
    "from_bert": False,
    "model": "Bert",
    "batch_size": 512,
    "sentence_length": 100,
    "learning_rate": 5e-4,
    "step_size": 5,
    "epoch": 100,
    "early_stop": True,
    "early_stop_steps": 10,
    "test_only": False,
    "ckpt": None
}

base_config = {
    "num_embeddings": 59290, #! Do not change
    "embedding_dim": 50, #! Do not change
    "hidden_size": 64,
    "pretrained_embedding_path": "/zhangpai21/workspace/zkw/semantic_diffusion_tasks/Sentiment Classification/Dataset/wiki_embedding.pth",
    "freeze_pretrained_embedding": False,
    "dropout_rate": 0.5,
    "init_config": {
       "method": "default"
    }
}

cnn_config = {
    "kernel_sizes": [3, 4, 5]
}

rnn_config = {
    "type": "LSTM",
    "num_layers": 2,
    "bidirectional": True
}

mlp_config = {
    "num_layers": 2
}

attn_config = {
    "num_heads": 5, #! Note: embedding_dim must be divisible by num_heads
    "num_layers": 2,
    "feature_size": 64,
    "extract_feature_method": "attn"
}

bert_config = {
    "freeze_bert": True,
    "pooling_type": "cls",
    "hidden_size": 768, #! Do not change
    "dropout_rate": 0.5
}