import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import Dict
from omegaconf import OmegaConf
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, base_conf: Dict) -> None:
        '''
        Args of base_config:
        - num_embeddings [int]
        - embedding_dim [int]
        - hidden_size [int]
        - sentence_length [int]
        - pretrained_embedding_path [str | None]
        - freeze_pretrained_embedding [bool]
        - dropout_rate [float]
        - init_config [Dict (must include key "method")]
        '''
        super().__init__()
        base_conf = OmegaConf.create(base_conf)
        self.num_embeddings = base_conf.num_embeddings
        self.embedding_dim = base_conf.embedding_dim
        self.hidden_size = base_conf.hidden_size
        self.sentence_length = base_conf.sentence_length
        if base_conf.pretrained_embedding_path:
            pretrained_embedding = torch.load(base_conf.pretrained_embedding_path)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=base_conf.freeze_pretrained_embedding)
        else:
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.dropout_rate = base_conf.dropout_rate
        self.init_config = base_conf.init_config
        
    def init_weights(self):
        method = self.init_config["method"]
        for n, p in self.named_parameters():
            if p.requires_grad:
                if method == "uniform":
                    nn.init.uniform_(p, self.init_config["a"], self.init_config["b"])
                elif method == "normal":
                    nn.init.normal_(p, 0, self.init_config["std"])
                else:
                    if len(p.shape) < 2:
                        eval(f"nn.init.{method}_")(p.unsqueeze(0))
                    else:
                        eval(f"nn.init.{method}_")(p)

class TextCNN(TextModel):
    def __init__(self, base_config: Dict, cnn_config: Dict):
        '''
        Args of cnn_config:
        - kernel_sizes [List[int]]
        '''
        super().__init__(base_config)
        cnn_conf = OmegaConf.create(cnn_config)
        self.__name__ = "TextCNN"
        self.kernel_sizes = cnn_conf.kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_size, kernel_size=kernel_size)
                for kernel_size in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.num_features = len(self.kernel_sizes)
        self.linear = nn.Linear(self.num_features * self.hidden_size, 2)
        if self.init_config["method"] != "default":
            self.init_weights()
        
    def forward(self, sentence, is_emb=False):
        if is_emb:
            emb = sentence.permute(0, 2, 1)
        else:
            emb = self.embedding(sentence).permute(0, 2, 1) # [bs, c_in (vector_size), len]
        conv_outs = [F.relu(conv(emb)) for conv in self.convs] # num_features * [bs, c_out, len_out]
        pool_outs = [F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in conv_outs] # num_features * [bs, c_out]
        feature = self.dropout(torch.cat(pool_outs, dim=1)) # [bs, num_features * c_out]
        out = self.linear(feature)
        return out
    
class TextRNN(TextModel):
    def __init__(self, base_config: Dict, rnn_config):
        '''
        Args of rnn_config:
        - type ["RNN" | "LSTM" | "GRU"]
        - num_layers [int]
        - bidirectional [bool]
        '''
        super().__init__(base_config)
        rnn_conf = OmegaConf.create(rnn_config)
        self.type = rnn_conf.type
        self.__name__ = f"TextRNN_{self.type}"
        self.num_layers = rnn_conf.num_layers
        self.bidirectional = rnn_conf.bidirectional
        self.rnn = eval(f"nn.{self.type}")(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size * (1 + int(self.bidirectional)), 2)
        if self.init_config["method"] != "default":
            self.init_weights()
        
    def forward(self, sentence, is_emb=False):
        if is_emb:
            emb = sentence
        else:
            emb = self.embedding(sentence)
        if self.type == "LSTM":
            _, (h_n, _) = self.rnn(emb)
        else:
            _, h_n = self.rnn(emb)
        h_n = h_n.transpose(0, 1)
        # h_n: [bs, num_layers * (1 + int(bidirectional)), hidden_size]
        if self.bidirectional:
            h_n = h_n.reshape(-1, self.num_layers, self.hidden_size * 2)
        h_n = h_n[:, -1]
        out = self.linear(h_n)
        return out
    
class TextMLP(TextModel):
    def __init__(self, base_config: Dict, mlp_config: Dict):
        '''
        Args of mlp_config:
        - num_layers [int]
        '''
        super().__init__(base_config)
        mlp_conf = OmegaConf.create(mlp_config)
        self.__name__ = "TextMLP"
        self.num_layers = mlp_conf.num_layers
        self.input_layer = nn.Linear(self.embedding_dim, self.hidden_size)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layers-1)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(self.hidden_size, 2)
        if self.init_config["method"] != "default":
            self.init_weights()
        
    def forward(self, sentence, is_emb=False):
        if is_emb:
            emb = sentence
        else:
            emb = self.embedding(sentence)
        x = F.relu(self.input_layer(emb))
        for layer in self.mlp_layers:
            x = x + F.relu(layer(x))
        x = x.permute(0, 2, 1)
        out = self.output_layer(self.dropout(F.max_pool1d(x, x.shape[2]).squeeze(2)))
        return out
    
class TextSelfAttention(TextModel):
    def __init__(self, base_config: Dict, attention_config: Dict):
        '''
        Args of attention_config:
        - num_heads [int]
        - num_layers [int]
        - feature_size [int]
        - extract_feature_method ["cnn" | "mlp" | "attn"]
            - "cnn": cnn_config
            - "rnn": mlp_config
        '''
        super().__init__(base_config)
        attn_conf = OmegaConf.create(attention_config)
        self.num_heads = attn_conf.num_heads
        self.num_layers = attn_conf.num_layers
        self.feature_size = attn_conf.feature_size
        self.mode = attn_conf.extract_feature_method
        self.__name__ = f"TextAttn_{self.mode}"
        self.positional_embedding = nn.Embedding(self.sentence_length, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        if self.mode == "cnn":
            self.output_layer = TextCNN(base_config, attn_conf.cnn_config)
        elif self.mode == "mlp":
            self.output_layer = TextMLP(base_config, attn_conf.mlp_config)
        elif self.mode == "attn":
            self.to_q = nn.Linear(self.embedding_dim, self.feature_size)
            self.to_k = nn.Linear(self.embedding_dim, self.feature_size)
            self.to_v = nn.Linear(self.embedding_dim, self.feature_size)
            self.scale = self.feature_size ** -0.5
            self.linear = nn.Linear(self.feature_size, 2)
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            assert False, "Unsupported extract feature method"
        self.register_buffer(
            "position",
            torch.tensor(range(self.sentence_length))
        )
        if self.init_config["method"] != "default":
            self.init_weights()

    def forward(self, sentence):
        emb = self.embedding(sentence) + self.positional_embedding(self.position)
        feature = self.encoder(emb)
        if self.mode == "cnn" or self.mode == "mlp":
            return self.output_layer(feature, is_emb=True)
        else:
            q, k, v = self.to_q(feature), self.to_k(feature), self.to_v(feature)
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn_map = sim.softmax(dim=-1)
            out = einsum("b i j, b j d -> b i d", attn_map, v)
            out = out.permute(0, 2, 1)
            out = self.linear(self.dropout(F.max_pool1d(out, out.shape[2]).squeeze(2)))
            return out
        
class BertBasedModel(nn.Module):
    def __init__(self, bert_config: Dict):
        '''
        Args of bert_config:
        - freeze_bert [bool]
        - pooling_type ["cls" | "pooler" | "last-avg" | "first-last-avg"]
        - hidden_size [int]
        - dropout_rate [float]
        '''
        super().__init__()
        bert_conf = OmegaConf.create(bert_config)
        self.__name__ = "BertBasedModel"
        self.freeze_bert = bert_conf.freeze_bert
        self.pooling = bert_conf.pooling_type
        self.hidden_size = bert_conf.hidden_size
        self.dropout_rate = bert_conf.dropout_rate
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        if self.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear = nn.Linear(self.hidden_size, 2)
        
    def forward(self, sentence):
        '''
        Honor Code: Partially refer to https://github.com/shuxinyin/Chinese-Text-Classification
        '''
        out = self.bert(sentence, output_hidden_states=True)
        
        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0, :]  # [bs, 768]
        elif self.pooling == 'pooler':
            out = out.pooler_output  # [bs, 768]
        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [bs, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [bs, 768]
        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [bs, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [bs, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [bs, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [bs, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [bs, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [bs, 768]
        else:
            assert False, "Unsupported pooling method"

        out = self.linear(self.dropout(out))
        return out
