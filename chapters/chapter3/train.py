
from chapter1.simplebooks import get_dataloaders, get_tokenizer, parent_path
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from chapter2.gptlikemodel import SLLM, SLLMConfig
import os






class LLMLoss(nn.Module):
    def __init__(self):
        super(LLMLoss, self).__init__()
    
    def forward(self, logits, targets):
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten())
        return loss

        
def batch_loss(loss_fn, input_batch,target_batch, model):

    assert model is not None
    assert input_batch is not None 
    assert target_batch is not None

    input_batch  = input_batch.to("cuda")
    target_batch = target_batch.to("cuda")
    
    with torch.no_grad():
        logits = model(input_batch)
        loss   = loss_fn(logits, target_batch)


    return loss

def loader_loss(loss_fn, data_loader, model):

    assert data_loader is not None
    assert model is not None

    total_loss = 0
    num_batches = len(data_loader)

    for i, batch in enumerate(data_loader):

        features, target = batch
        loss = batch_loss(loss_fn, features, target, model)
        total_loss+=loss
        

    return total_loss / num_batches



def generate_text(model, idx, max_new_tokens, context_size):
    """
    Generate output tokens from a given model.
    Arguments:
        model: 
            llm model for text generation
        idx:
            Input token tensor
        max_new_tokens:
            Number of output tokens to be generated
        context_size:
            model context window.
    """
    for _ in range(max_new_tokens):
        idx_trim = idx[:,-context_size:]
        
        with torch.no_grad():
            logits = model(idx_trim)
        
        logits = logits[:,-1,:]
        probas = torch.softmax(logits, dim=-1)
        
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def invoke_model(model, start_context):
    
    assert len(start_context) > 0 \
        and start_context is not None
        
    print(f"Input context: '{start_context}'\n")
    tokenizer = get_tokenizer()
    encoded = tokenizer.encode(start_context)
    
    # convert to tensor and add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"Encoded tensor {encoded_tensor} No Tokens: {encoded_tensor.size()[-1]} \n")
    
    model.eval()
    encoded_tensor = encoded_tensor.to("cuda")
    with torch.no_grad():
        out = generate_text(model, encoded_tensor, 5, context_size=50)
    print(f"Output {out} No Tokens: {out.size()[-1]}")
    
    decoded_text = tokenizer.decode(out.squeeze(0))
    print(f"Decoded text: '{decoded_text}'")

    
# Load train,validation and test datasets
train_loader, valid_loader, test_loader = get_dataloaders(batch_size=64, \
                num_workers=4)

# Model
# Initialize the model class
config = SLLMConfig()

print(f"Model configuration {config}\n")

model = SLLM(config)
save_directory = parent_path + "/bin/"

## Learning rate warmup
start_context = "wonderful spring is awaited."

n_epochs = 100
initial_lr = 1e-4
min_lr = 1e-7
top_lr = 0.01
warmup_steps = 2500
total_training_steps = n_epochs * len(train_loader)
device = "cuda"
progress_bar = tqdm(range(total_training_steps))
eval_freq = 5000

lr_increment = (top_lr - initial_lr) / warmup_steps


optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
loss_fn = LLMLoss()

global_steps = -1
tokens_seen = 0

track_lrs = []

train_losses = []
eval_losses  = []
prev_eval_loss = 1e5
model = model.to(device)

for epoch in range(n_epochs):
    
    losses = []
    model.train()
    for input_batch in train_loader:
        
        features, target = input_batch
        features = features.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        global_steps+=1
        
        if global_steps < warmup_steps:
            lr = initial_lr + global_steps * lr_increment
        else:
            # cosine decay
            progress = (global_steps - warmup_steps) / (total_training_steps - warmup_steps)
            lr = min_lr + (top_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
        
        for param_group in optimizer.param_groups:
            param_group["lr"] =lr
        
        logits = model(features)
        loss = loss_fn(logits, target)
        
        tokens_seen += features.numel()
        
        loss.backward()
        
        if global_steps > warmup_steps:
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
        losses.append(loss.item())
        track_lrs.append(lr)
        
        progress_bar.update(1)
        
        
        if global_steps % eval_freq == 0:
            model.eval()
            eval_loss = loader_loss(loss_fn, valid_loader, model)
            model.train()
            print(f"Epoch {epoch} Evaluation Loss {eval_loss} LR {lr}")
            
            if eval_loss < prev_eval_loss:
                print(f"saving model checkpoint")
                file_name = f"small_llm-v1-{epoch}-{eval_loss:.3}"
                torch.save(model.state_dict(), save_directory + file_name)
                prev_eval_loss = eval_loss
                
                
            eval_losses.append((epoch, eval_loss))
        
        
        
    
    print(f"Epoch {epoch} Avg Train Loss {sum(losses)/len(losses)} LR {lr}")
    invoke_model(model, start_context)

torch.save(model.state_dict(), save_directory + "small_llm-v1.pt")

