from datasets import load_dataset


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
from pathlib import Path
import sys



current_path = os.path.abspath(__file__)
parent_path    = str(Path(current_path).parent.parent.parent.absolute())

print(current_path, parent_path)

def get_tokenizer():
    
    tokenizer_path = parent_path + "/data/simplebooks-tokenizer"
    print(f"Loading tokenizer from {tokenizer_path}")
    simplebooks_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    return simplebooks_tokenizer


def download_simplebooks(destination: str) -> None:
    """
    Download simple books dataset
    
    Args:
        destination: download folder string
    """
    url = "https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip"
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path="destination")
    print(f"Finished downloading and extracting simplebooks.zip")


def load_simple_books(destination: str) -> dict:
    """
    
    """
    dataset = load_dataset(destination)
    
    train        = dataset['train']['text']
    test         = dataset['test']['text']
    validation   = dataset['validation']['text'] 
    
    return {"train": train, "test": test, "validation": validation}
                                         



class SimpleBooksDataSet(Dataset):
    """

    """
    def __init__(self, corpus, max_length, stride, context='train',tokenizer_path=None):
        """

        """
        path = tokenizer_path
        if tokenizer_path == None:
            path = parent_path + "/data/simplebooks-tokenizer"
        
        self.tokenizer  = AutoTokenizer.from_pretrained(path)
        print(f"Loading tokenizer from {path}")
        self.input_ids  = []
        self.target_ids = []
        self.token_ids  = []

        for sample in corpus:
            if len(sample) > 0:
                self.token_ids.extend(self.tokenizer.encode(sample, truncation=True, max_length=max_length))
        
        print(f"Total {context} tokens {len(self.token_ids):,}")
        
        for i in range(0, len(self.token_ids) - max_length + 1,stride):
            input_chunk =  self.token_ids[i:i + max_length]
            target_chunk = self.token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


    
def stack_collate(data):
    features, target = zip(*data)
    X = torch.stack(features)
    y = torch.stack(target)
    return X, y
    
def get_dataloaders(batch_size=64, num_workers=4,tokenizer_path=None):
    """
    
    """

    
    destination = parent_path + '/data/simplebooks/simplebooks-2-raw/'
    
    print(f"Loading dataset from {destination}")
    
    dataset     = load_simple_books(destination)
    
    training_corpus   = dataset['train']
    validation_corpus = dataset['validation']
    test_corpus       = dataset['test']

    
    train_ds      = SimpleBooksDataSet(training_corpus, max_length=50, stride=5,tokenizer_path=tokenizer_path)            
    validation_ds = SimpleBooksDataSet(validation_corpus,max_length=50, stride=5, context='validation',tokenizer_path=tokenizer_path)  
    test_ds = SimpleBooksDataSet(test_corpus,max_length=50, stride=5, context='test',tokenizer_path=tokenizer_path)  


    train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, collate_fn=stack_collate,
                                  shuffle=True,drop_last=True,num_workers=num_workers)

    validation_dataloader = DataLoader(dataset=validation_ds, batch_size=batch_size, collate_fn=stack_collate,
                                  shuffle=False,drop_last=True,num_workers=num_workers)
    
    test_dataloader = DataLoader(dataset=validation_ds, batch_size=batch_size, collate_fn=stack_collate,
                                  shuffle=False,drop_last=True,num_workers=num_workers)

    
    
    return train_dataloader, validation_dataloader, test_dataloader

                             
