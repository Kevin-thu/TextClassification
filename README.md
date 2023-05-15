# Sentiment Classification

This code provides a sentiment classification model for text data. The model can be trained on a dataset of labeled text data and then used to predict the sentiment of new text data. There are 5 types of models provided: TextCNN, RNN (including Vanilla RNN, LSTM, GRU), MLP, Self Attention and Bert-based model.

## How to Run

To run the code, first set your configurations in config.py. You can also set some frequently used configurations by terminal arguments. The available arguments are:

- -m or --model: the type of model to use (CNN, RNN, MLP, Attn, or Bert)
- -lr or --learning-rate: the learning rate for the optimizer
- -e or --epoch: the number of epochs to train for
- -b or --batch-size: the batch size for training
- -hs or --hidden-size: the hidden size for the model (only applicable for some models)
- -t or --test: whether to only run testing (no training)

Once your configurations are set, you can run the code with the command:

```
python main.py
```

## Dependencies

This code requires the following dependencies:

- torch
- tqdm
- wandb
- pathlib
- sklearn
- omegaconf

You can install these dependencies with the command:
```
pip install -r requirements.txt
```

Datasets and current trained sota checkpoitns have been uploaded to https://cloud.tsinghua.edu.cn/d/5489da1cb5c3476e8336/. For more information on this project, please read `REPORT.pdf`.