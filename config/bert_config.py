train_config = {
    "seed": 999,
    "from_bert": True,
    "model": "Bert",
    "batch_size": 64,
    "sentence_length": 100,
    "learning_rate": 1e-3,
    "epoch": 100,
    "step_size": None,
    "early_stop": True,
    "early_stop_steps": 10,
    "test_only": False,
    "ckpt": None
}

bert_config = {
    "freeze_bert": True,
    "pooling_type": "cls",
    "hidden_size": 768,  # Do not change
    "dropout_rate": 0.5
}
