import torch


def greedy_search(**kwargs):
    logits = kwargs['logits']
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    return idx_next

def probabilistic_search(**kwargs):
    logits = kwargs['logits']
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probas, num_samples=1)
    return idx_next

def temperature_scaling(**kwargs):
    logits = kwargs['logits']
    temperature = kwargs['temperature']
    probas = torch.softmax(logits/temperature, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    return idx_next

def generate_text(model, idx, max_new_tokens
                  , context_size
                  , search_fn=greedy_search
                  , temperature=1.0):
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
        idx_next = search_fn(logits=logits,temperature=temperature)
        
        idx = torch.cat((idx, idx_next), dim=1)
    return idx



def invoke_model(model,tokenizer 
                 ,start_context
                 ,search_fn=greedy_search
                ,temperature=1.0):
    
    assert len(start_context) > 0 \
        and start_context is not None
        
    print(f"Input context: '{start_context}'")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = generate_text(model, encoded_tensor, 5
                            , context_size=50
                            ,search_fn=search_fn
                           ,temperature=temperature)
    
    decoded_text = tokenizer.decode(out.squeeze(0))
    print(f"Decoded text: '{decoded_text}'\n")