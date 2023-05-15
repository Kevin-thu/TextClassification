import os, random
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
from sklearn.metrics import f1_score
from omegaconf import OmegaConf
from src.data_process import SentimentDataset
from src.models import TextCNN, TextRNN, TextMLP, TextSelfAttention, BertBasedModel
from config.config import train_config, base_config, cnn_config, rnn_config, mlp_config, attn_config, bert_config
"""
Model configs of provided ckpt:

from config.cnn_config import train_config, base_config, cnn_config
from config.rnn_config import train_config, base_config, rnn_config
from config.mlp_config import train_config, base_config, mlp_config
from config.attn_config import train_config, base_config, attn_config
from config.bert_config import train_config, bert_config
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", type=str, choices=["CNN", "RNN", "MLP", "Attn", "Bert"])
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("-e", "--epoch", dest="epoch", type=int)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("-hs", "--hidden-size", dest="hidden_size", type=int)
    parser.add_argument("-t", "--test", dest="test", action="store_true")
    args = parser.parse_args()
    return args.model, args.learning_rate, args.epoch, args.batch_size, args.hidden_size, args.test

def get_dataloader(batch_size, sentence_length, from_bert=False):
    
    train_dataset = SentimentDataset("train", sentence_length=sentence_length, from_bert=from_bert)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)

    val_dataset = SentimentDataset("validation", sentence_length=sentence_length, from_bert=from_bert)
    val_dataloader=DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)

    test_dataset = SentimentDataset("test", sentence_length=sentence_length, from_bert=from_bert)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2) 

    return train_dataloader, val_dataloader, test_dataloader

def train(model, epoch, batch_size, dataloader, optimizer, scheduler):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    iterator = tqdm(dataloader, desc=f"Epoch {epoch}")
    train_loss, train_acc = 0.0, 0.0
    correct_count, total_count = 0, 0
    y_true, y_pred = [], []
    for x, y in iterator:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss_func.zero_grad()
        out = model(x)
        pred = out.argmax(1)
        loss = loss_func(out, y)
        iterator.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct_count += (y == pred).float().sum().item()
        total_count += len(y)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    train_loss = train_loss * batch_size / len(dataloader.dataset)
    train_acc = correct_count / total_count
    train_fs = f1_score(y_true, y_pred, average="binary")
    if scheduler:
        scheduler.step()
    return train_loss, train_acc, train_fs

def val_or_test(mode, model, batch_size, dataloader):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    iterator = tqdm(dataloader, desc=f"{mode}")
    test_loss, test_acc = 0.0, 0.0
    correct_count, total_count = 0, 0
    y_true, y_pred = [], []
    for x, y in iterator:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss_func.zero_grad()
        out = model(x)
        pred = out.argmax(1)
        loss = loss_func(out, y)
        iterator.set_postfix(loss=loss.item())
        test_loss += loss.item()
        correct_count += (y == pred).float().sum().item()
        total_count += len(y)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    test_loss = test_loss * batch_size / len(dataloader.dataset)
    test_acc = correct_count / total_count
    test_fs = f1_score(y_true, y_pred, average="binary")
    return test_loss, test_acc, test_fs

if __name__ == "__main__":
    '''
    Note: Please set your configurantions in `config.py` before training / testing.
        You can also set some frquently used configurations by terminal arguments. 
    
    Args of train_config:
    - seed [int]
    - from_bert [bool]
    - model ["CNN" | "RNN" | "MLP" | "Attn" | "Bert"]
    - batch_size [int]
    - sentence_length [int]
    - epoch [int]
    - learning_rate [float]
    - step_size: [int | None]
    - early_stop [bool]
    - early_stop_steps [int]
    - test_only [bool]
    - ckpt [str | None]
    '''
    conf = OmegaConf.create(train_config)
    m, lr, e, bs, h, t = parse_args()
    if m: conf.model = m
    if lr: conf.learning_rate = lr
    if e: conf.epoch = e
    if bs: conf.batch_size = bs
    if h: base_config["hidden_size"] = conf.hidden_size
    if t: conf.test_only = t
    if conf.model != "Bert":
        base_config["sentence_length"] = conf.sentence_length
    
    random.seed(conf.seed)
    os.environ["PYTHONHASHSEED"] = str(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)

    if conf.model == "CNN":
        model = TextCNN(base_config, cnn_config)
    elif conf.model == "RNN":
        model = TextRNN(base_config, rnn_config)
    elif conf.model == "MLP":
        model = TextMLP(base_config, mlp_config)
    elif conf.model == "Attn":
        model = TextSelfAttention(base_config, attn_config)
    elif conf.model == "Bert":
        conf.from_bert = True
        model = BertBasedModel(bert_config)
    else:
        assert False, "Unsupported model type"
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(conf.batch_size, conf.sentence_length, conf.from_bert)
    
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, conf.step_size) if conf.step_size else None
    
    """For visualization:
    
    wandb.init(
        project="IAI 2023 Sentiment Classification",
        name=f"{model.__name__}",
        entity="kevin-thu",
        config = {
            "learning_rate": conf.learning_rate,
            "epochs": conf.epoch,
            "batch_size": conf.batch_size
        }
    )
    """

    CKPT_PATH = Path(__file__).parent / "ckpt"
    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    if not os.path.exists(CKPT_PATH / f"{model.__name__}"):
        os.makedirs(CKPT_PATH / f"{model.__name__}")
    if conf.ckpt:
        save_path = CKPT_PATH / f"{model.__name__}" / conf.ckpt
        ckpt = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    else:
        save_path = CKPT_PATH / f"{model.__name__}" / f"lr_{conf.learning_rate}_bs_{conf.batch_size}.ckpt"
    print("Save path:", save_path)
        
    if not conf.test_only:
        es_count = 0
        _, _, best_fs = val_or_test("Val", model, conf.batch_size, val_dataloader)
        print("------Training------")
        print("Device:", DEVICE)
        print(f"Val f1score starting from: {best_fs}")
        
        for epoch in range(conf.epoch):
            train_loss, train_acc, train_fs = train(
                model,
                epoch,
                conf.batch_size,
                train_dataloader,
                optimizer,
                scheduler
            )
            val_loss, val_acc, val_fs = val_or_test(
                "Val",
                model,
                conf.batch_size,
                val_dataloader
            )
            """ 
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_fs": train_fs,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_fs": val_fs
            }) 
            """
            if not conf.early_stop:
                print("Val f1score:", val_fs)
            else:
                if val_fs <= best_fs:
                    print("Val f1score:", val_fs)
                    es_count += 1
                    if es_count >= conf.early_stop_steps:
                        break
                else:
                    print("Val f1score*:", val_fs)
                    es_count = 0
                    best_fs = val_fs
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "fs": best_fs
                    }
                    torch.save(ckpt, save_path)

    if conf.early_stop:
        ckpt = torch.load(save_path)
        model.load_state_dict(ckpt["state_dict"])
    print("------Testing------")
    if conf.early_stop:
        print("Epoch:", ckpt["epoch"])
        print("Val f1score:", ckpt["fs"])
    test_loss, test_acc, test_fs = val_or_test("Test", model, conf.batch_size, test_dataloader)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Test f1score:", test_fs)
