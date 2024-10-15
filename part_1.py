
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


train_batch_size = 16  # training batch size
eval_batch_size = 8  # evaluation batch size
context_length = 256  # number of tokens processed in a single batch
train_split = 0.8  # percentage of data to use from total data for training

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
            d = torch.cat([d, self.tokens[:add_data]])
        x = (d[:-1]).view(b, c)  # inputs
        y = (d[1:]).view(b, c)  # targets

        self.current_position += b * c # set the next position
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


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
    
    def forward(self, inputs, targets = None):
        logits = self.wte(inputs) # dim -> batch_size, sequence_length, d_model
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            # to calculate loss for all token embeddings in a batch
            # kind of a requirement for cross_entropy
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
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

m = GPT(vocab_size=vocab_size, d_model=d_model).to(device)


# We have now successfully defined our model with just one Embedding layer and Softmax for token generation.
# Let's see how our model behaves when given some input characters.

with torch.no_grad():
    input = torch.tensor(encode("Love"), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(m.generate(input, max_new_tokens=144)[0].numpy()))

#####################################################################################

lr = 5e-4
optim = torch.optim.AdamW(m.parameters(), lr=lr)

epochs = 2000
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
