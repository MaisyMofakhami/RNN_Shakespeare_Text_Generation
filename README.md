"love is not my true love , and i am less than a king . and tell you , my lord , i will have you." : GRU Model ;)

# Shakespeare-Like Text Generation Using GRU

This project implements a neural text generation system trained on the Tiny Shakespeare dataset using GRU-based recurrent neural networks. The work was completed as part of MGSC695 - Deep Learning for NLP.

## 📂 Dataset

- **Source**: Hugging Face Datasets
- **Name**: `karpathy/tiny_shakespeare`
- **Size**: Approximately 1MB
- **Format**: Plain text containing lines from Shakespeare’s plays

## 🔧 Preprocessing

Text preprocessing involved the following steps:
- Loading the dataset from Hugging Face
- Tokenization using BERT’s WordPiece tokenizer (`bert-base-uncased`)
- Optional lemmatization using `nltk`'s `WordNetLemmatizer`
- Removal of stopwords
- Vocabulary mapping using `stoi` and `itos`
- Creation of sequences using a sliding window approach with fixed `seq_len`

## 🧠 Model Architecture

The primary model used is a bidirectional GRU-based RNN:

```python
class GRUModel_Bi(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        ...
```

- **Embedding Layer**: `nn.Embedding`
- **Recurrent Layer**: `nn.GRU` (bidirectional)
- **Output Layer**: Fully connected (`nn.Linear`)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam or RMSprop
- **Gradient Clipping**: Applied to prevent exploding gradients

## 🏋️‍♀️ Training

The model is trained using teacher forcing. The dataset is split using a `split_data_tokens` function which separates the dataset into training, validation, and test sets (80/10/10).

Hyperparameters include:
- `embedding_dim`: 86 or 128
- `hidden_dim`: 64 or 128
- `seq_len`: 50
- `batch_size`: 64
- `epochs`: 10
- `optimizer`: Adam / RMSprop/ SGD

Best model performance was observed using:
- GRU on token-based dataset
- `embedding_dim=86`, `hidden_dim=128`, optimizer=`RMSprop`

## 🧪 Evaluation

- **Metric**: Perplexity and CrossEntropyLoss
- Test evaluation was done using a held-out test set.
- Loss and perplexity were calculated using `evaluate_model()`:
  ```python
  loss = criterion(output.view(-1, vocab_size), y.view(-1))
  perplexity = torch.exp(torch.tensor(loss))

## 📊 Model Metrics

✅ BEST CONFIG FOR DF_TOKENS: embed=86, hidden=128, opt=RMSPROP | Val Loss: 0.4363, Perplexity: 1.55

✅ BEST CONFIG FOR DF_LONG: embed=86, hidden=128, opt=RMSPROP | Val Loss: 0.7485, Perplexity: 2.11


## ✍️ Text Generation

Text is generated using temperature-controlled sampling. The function `generate_text_rnn()` takes a seed prompt and generates text of specified length.

- **Temperature < 1**: More deterministic and repetitive
- **Temperature = 1**: Balanced creativity
- **Temperature > 1**: More random and diverse outputs

Sample command:
```python
generate_text_rnn(model, "love is not", temperature=1.0, gen_length=50)
```

## 📃 Saved Models

The following models were saved after training:
- `GRU_Teacher_Forcing_model_tokens.pth`
- `GRU_Teacher_Forcing_model_long.pth`
