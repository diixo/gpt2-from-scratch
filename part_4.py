
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
gen_size = 200          # should be less than context_length

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
# InputTokens -->> Embedding -->> Softmax -->> Prediction

# used to define size of embeddings
d_model = vocab_size

# The embedding dimension or d_model is vocab_size currently because the final output has to map 
# to the logits for each character in vocab to calculate their probabilities. 
# Later on we will introduce a Linear layer which will map d_model to vocab_size and then 
# we can have a custom embedding_dimension.


# define our PE Class
class PositionalEncoding(nn.Module):

    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # Create a matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)
        
        # Create a vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        
        # Create a vector with the divisor terms based on the dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)
        
        # Register pe as a buffer, so it is not considered a parameter but is part of the module's state
        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encodings to the input embeddings
        return x + self.pe[:,:x.size(1), :]



n_heads = 4 # number of self-attention heads. should be divisible with d_model (d_model=vocabulary_size=92)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert (n_heads * self.head_dim == d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)


    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape
        
        # Project the input embeddings into Q, K, and V
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenate heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        # Apply the final linear transformation
        out = self.fc_out(attention_output)
        
        return out


class GPT(nn.Module):

    def __init__(self, vocab_size, d_model, n_heads):

        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
        # initialize positional encodings
        self.wpe = PositionalEncoding(context_length, d_model)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.att = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        #self.linear1 = nn.Linear(d_model, vocab_size)


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        batch_size, sequence_length = inputs.shape

        logits = self.wte(inputs)
        logits = self.wpe(logits)
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        #logits = self.linear1(logits)

        loss = None
        if targets is not None:
            # Преобразование логитов и целей для корректного вычисления кросс-энтропии
            logits = logits.view(batch_size * sequence_length, logits.size(-1))  # [batch_size * seq_length, d_model]
            targets = targets.view(batch_size * sequence_length)  # [batch_size * seq_length]

            # Вычисление функции потерь
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, inputs, max_new_tokens):
        # this will store the model outputs along with the initial input sequence
        # make a copy so that it doesn't interfare with model 
        for _ in range(max_new_tokens):
            # we only pass targets on training to calculate loss
            logits, _ = self(inputs)  
            # for all the batches, get the embeds for last predicted sequence
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=1)            
            # get the probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            inputs = torch.cat([inputs, idx_next], dim=1)
        # as the inputs has all model outputs + initial inputs, we can use it as final output
        return inputs


m = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads).to(device)


# We have now successfully defined our model with just one Embedding layer and Softmax for token generation.
# Let's see how our model behaves when given some input characters.

with torch.no_grad():
    input = torch.tensor(encode("Love"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(m.generate(input, max_new_tokens=gen_size)[0].numpy()))

#####################################################################################

lr = 1e-3
optim = torch.optim.AdamW(m.parameters(), lr=lr)

epochs = 5000
eval_steps = 100 # perform evaluation in every n steps
for ep in range(epochs):
    xb, yb = train_loader.get_batch()

    logits, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if ep % eval_steps == 0 or ep == epochs-1:
        m.eval()
        with torch.no_grad():
            xvb, yvb = eval_loader.get_batch()
            _, e_loss = m(xvb, yvb)

            print(f"Epoch: {ep}\tlr: {lr}\ttrain_loss: {loss}\teval_loss: {e_loss}")
        m.train() # back to training mode


with torch.no_grad():
    input = torch.tensor(encode("Love"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(m.generate(input, max_new_tokens=gen_size)[0].numpy()))
