import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# use cpu or gpu based on your system
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


data_dir = "data.txt"
text = open(data_dir, 'r').read() # load all the data as simple string

# Get all unique characters in the text as vocabulary
chars = list(set(text))
vocab_size = len(chars)

# build the character level tokenizer
chr_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_chr = {i:c for i, c in enumerate(chars)}

def encode(input_text: str) -> list[int]:
    return [chr_to_idx[t] for t in input_text]

def decode(input_tokens: list[int]) -> str:
    return "".join([idx_to_chr[i] for i in input_tokens])


# convert our text data into tokenized tensor
data = torch.tensor(encode(text), dtype=torch.long, device=device)


train_batch_size = 16   # training batch size
eval_batch_size = 8     # evaluation batch size
context_length = 256    # number of tokens processed in a single batch
train_split = 0.8       # percentage of data to use from total data for training
gen_size = 180          # should be less than context_length

# split data into trian and eval
n_data = len(data)
train_data = data[:int(n_data * train_split)]
eval_data = data[int(n_data * train_split):]


class DataLoader:
    def __init__(self, tokens, batch_size, context_length) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length

        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length

        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        # if the batch exceeds total length, get the data till last token
        # and take remaining from starting token to avoid always excluding some data
        add_data = (-1) # n, if length exceeds and we need `n` additional tokens from start
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens) - 1
            end_pos = len(self.tokens) - 1

        d = self.tokens[start_pos:end_pos]
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data + 2]])
            self.current_position = 0
        else:
            self.current_position += b * c # set the next position
        x = (d[:-1]).view(b, c)  # inputs
        y = (d[1:]).view(b, c)  # targets

        return x, y

train_loader = DataLoader(train_data, train_batch_size, context_length)
eval_loader = DataLoader(eval_data, eval_batch_size, context_length)


# Now we have our own customized data loader for both training and evaluation.
# The loader has a get_batch function which returns batches of batch_size * context_length.
xb, yb = train_loader.get_batch()
print(xb.shape, yb.shape)

##########################################################################################

