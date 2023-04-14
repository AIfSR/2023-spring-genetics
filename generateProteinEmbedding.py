#Dependencies
# !pip install pytorch-nlp
# !pip install transformers
# !pip install pytorch-lightning
# !pip install sentencepiece

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

def generate_embeddings(model_link, sequence_examples, per_protein= True, per_residue=False,):
  
  #Loading encoder model from URL
  print("Loading: {}".format(model_link))
  model = T5EncoderModel.from_pretrained(model_link)  
  model.full() if device=='cpu' else model.half() 
  model = model.to(device)
  model = model.eval()

  #Loading Tokenizer
  tokenizer = T5Tokenizer.from_pretrained(model_link, do_lower_case=False)

  #Preprocessing sequences - replace rare amino acids (UZOB) by X and introduce white-space between all amino acids
  sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

  #Generating tokens for the sequence - tokenize sequences and pad up to the longest sequence in the batch
  ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest") # this padding option is great/convenient
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  # generate embeddings
  with torch.no_grad():
    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

  embeddings_per_protein = []
  embeddings_per_residue = []
  for i in range(len(sequence_examples)):
    if per_protein:
      embeddings_per_protein.append(embedding_repr.last_hidden_state[0,:len(sequence_examples[i])].mean(dim=0))

    if per_residue:
      embeddings_per_residue.append(embedding_repr.last_hidden_state[0,:len(sequence_examples[i])])

  return embeddings_per_protein, embeddings_per_residue


# #Testing the function with dummy data
# sequence_examples = ["PRTEINO", "SEQWENCE"]
# transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
# embeddings_per_protein,embeddings_per_residue = generate_embeddings(transformer_link,sequence_examples)
# embeddings_per_protein