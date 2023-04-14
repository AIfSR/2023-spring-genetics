from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

from generateProteinEmbedding import *
from tsne import *

#Loading data from Fasta File
sequences =[]
for record in SeqIO.parse('samples.fasta','fasta'):
	sequences.append(str(record.seq))


#creating embeddings
embeddings_per_protein,embeddings_per_residue = generate_embeddings(transformer_link,sequences)

#storing them in memory
embeddings = []
for s in range(len(sequences)):
	embeddings.append(embeddings_per_protein[s].cpu().numpy())


#Saving Embeddings
np.savetxt('embedding.txt',np.array(embeddings))


#Dimensionality Reduction for Visualization in 2D
reduced_embeddings = tsne_transform(np.array(embeddings))
plt.scatter(reduced_embeddings[:,0],reduced_embeddings[:,1])