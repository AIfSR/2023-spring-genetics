from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from generateProteinEmbedding import *
from tsne import *

transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

#Loading data from Fasta File
# sequences =[]
# for record in SeqIO.parse('samples.fasta','fasta'):
# 	sequences.append(str(record.seq))


#Loading TE data
# te_df=pd.read_csv('T_vaginalis_G3.mavericks_for_ML.txt', sep='\t')
# te_df_filtered=te_df[te_df.length<=5000]
# sequences=te_df_filtered['sequence'].values


#Loading Sample PFAM data from csv
df = pd.read_csv('./data/sample_pfam')

#creating family label - Specific to pfam data
df['family_accession']=df.family_accession.map(lambda x : x[2:7])

#Filtering top 5 occuring labels -> 149 records - Specific to pfam data
most_common_families = df.family_accession.value_counts()[:5].index
df_sub = df[df.family_accession.isin(most_common_families.values)]
sequences = df_sub.sequence.values
labels = df_sub.family_accession.values


#Creating Embeddings
embeddings=[]
output_hidden_states=False  #Set true if you want to concatenate embeddings of last 4 hidden layers along with the last layer
num_hidden_states=4  #number of hidden layers retrieved 
batch_size = 32
iterations= len(sequences)//batch_size

# embeddings_per_protein_np=[]

for s in range(iterations):
  embeddings_per_protein,embeddings_per_residue, hidden_states = generate_embeddings(transformer_link,sequences[s*batch_size:(s+1)*batch_size],output_hidden_states, num_hidden_states)
  
  for i in range(batch_size):
    embeddings.append(embeddings_per_protein[i].cpu().numpy())
  
  # final_sequence = embeddings_per_protein
  
  # if output_hidden_states:  
  #   for i in range(num_hidden_states):
  #     hidden_states[i]=hidden_states[i].cpu().numpy()
  #   final_sequence = np.concatenate(hidden_states)
  #   final_sequence = np.concatenate([final_sequence, embeddings_per_protein])
  
  # embeddings.append(final_sequence)
  print("Step ",s,"Completed")

#Saving Embeddings
np.savetxt('embedding.txt',np.array(embeddings))


#Dimensionality Reduction for Visualization in 2D
reduced_embeddings = tsne_transform(np.array(embeddings))

#Visualizing PFAM embeddings with labels
for protein in most_common_families:
  filtered_df=df_sub[df.family_accession==protein]
  plt.scatter(filtered_df.comp1,filtered_df.comp2, label=protein)

plt.legend()
plt.title("Visualizing Embeddings of 149 protein sequences belonging to 5 classes")
plt.savefig("PFAM_Visualization.png")