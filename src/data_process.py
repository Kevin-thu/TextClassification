import os
import torch
import gensim
import pickle as pk
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

DATA_PATH = Path(__file__).parent.parent / "Dataset"

class SentimentDataProcessor:
    def __init__(self, data_path: Path = DATA_PATH, from_bert: bool = False):
        if from_bert:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        else:
            self.data_path = data_path
            if os.path.exists(data_path / "tokenizer.pk"):
                with open(data_path / "tokenizer.pk", "rb") as f:
                    self.tokenizer = pk.load(f, encoding="utf8")
            else:
                self.tokenizer = self.get_tokenizer()
            if os.path.exists(data_path / "wiki_embedding.pth"):
                with open(data_path / "wiki_embedding.pth", "rb") as f:
                    self.embedding = torch.load(data_path / "wiki_embedding.pth")
            else:
                self.tokenizer = self.get_embedding()
            print("num_embedding: ", len(self.tokenizer))
        
    def get_tokenizer(self) -> Dict[str, int]:
        tokenizer = {"": 0}
        for file in os.listdir(self.data_path):
            if file.endswith("txt"):
                with open(self.data_path / file, "r", encoding="utf8", errors="ignore") as f:
                    for line in tqdm(f.readlines(), desc=f"Get tokenizer from {file}"):
                        words = line.strip().split()[1:]
                        for word in words:
                            if word not in tokenizer:
                                tokenizer[word] = len(tokenizer)
        with open(self.data_path / "tokenizer.pk", "wb") as f:
            pk.dump(tokenizer, f)
        return tokenizer
    
    def get_embedding(self) -> torch.TensorType:
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.data_path / "wiki_word2vec_50.bin", binary=True)
        embedding = torch.zeros((len(self.tokenizer), word2vec_model.vector_size))
        for word, id in tqdm(self.tokenizer.items(), desc="Get embedding"):
            try:
                embedding[id] = torch.from_numpy(word2vec_model[word])
            except KeyError:
                pass
        torch.save(embedding, self.data_path / "wiki_embedding.pth")
        return embedding
                

class SentimentDataset(Dataset):
    def __init__(self, mode: str, data_path: Path = DATA_PATH, sentence_length: int = 60, from_bert: bool = False):
        super().__init__()
        self.data_path = data_path
        self.sentence_length = sentence_length
        self.from_bert = from_bert
        data_processor = SentimentDataProcessor(data_path, from_bert)
        self.tokenizer = data_processor.tokenizer
        filename = f"{mode}.pk" if not from_bert else f"{mode}_bert.pk"
        # if os.path.exists(data_path / filename):
        #     with open(data_path / filename) as f:
        #         self.data = pk.load(f, encoding="utf8")
        # else:
        self.data = self.load_data(mode)
        
    def load_data(self, mode: str):
        with open(self.data_path / f"{mode}.txt", "r", encoding="utf8", errors="ignore") as f:
            lines = f.readlines()
            data_size = len(lines)
            labels, sentences = torch.zeros(data_size, dtype=int), torch.zeros((data_size, self.sentence_length), dtype=int) 
            for idx, line in enumerate(tqdm(lines, desc=f"Load {mode} data")):
                line = line.strip().split()
                label, words = line[0], line[1:self.sentence_length+1]
                labels[idx] = torch.tensor(int(label))
                if self.from_bert:
                    sentence = "".join(words)
                    words = self.tokenizer(sentence).input_ids[:self.sentence_length]
                    words += [0] * max(self.sentence_length - len(words), 0)
                    sentences[idx] = torch.tensor(words, dtype=int)
                else:
                    words += [""] * max(self.sentence_length - len(words), 0)
                    sentences[idx] = torch.tensor(list(map(lambda word: self.tokenizer[word], words)), dtype=int)
        data = {
            "labels": labels,
            "sentences": sentences
        }
        # filename = f"{mode}.pk" if not self.from_bert else f"{mode}_bert.pk"
        # with open(self.data_path / filename, "wb") as f:
        #     pk.dump(data, f)
        return data
    
    def __len__(self):
        return len(self.data["labels"])
    
    def __getitem__(self, index):
        return (self.data["sentences"][index], self.data["labels"][index])
    
if __name__ == "__main__":
    train_dataset = SentimentDataset("train")
    validation_dataset = SentimentDataset("validation")
    test_dataset = SentimentDataset("test")