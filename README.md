# 2023-spring-genetics

This contains the code for models the team has built for solving the TE detection problem.

Contents:

## unet_mav_detector.ipynb
A unet model architecture whch attempts to locate the maverick type TE element in a genome sequence. The first part of the code chunks a DNA sequence 
into strings of length 30k, each containing a maverick TE at a random location within that chunk. The string is converted into a 2-d one-hot encoding
where each of the 4 arrays represent one of the four possible base elements (A, G, C, and T). Finally a mask is created to represent where in the genome
the maverick TE is located. The model itself is based on the U-NET architecture. The model has three "contraction" blocks, each of which consists of two 
convolutional layers followed by a batch normalization and then a relu activation function for non-linearity. At the end of each contraction block, max 
pooling is applied to pool the features and produce a smaller output. This is followed by three expansion blocks, each with a convolutional transpose 
layer. Binary cross-entropy is used as the loss algorithm and IoU as the metric. 
The model did learn to estimate the rough location of the TE. However, IoU is not achieving more than 0.7 at best and this model is very slow.

## protT5_classifier.ipynb (WIP)
The purpose of this file is to create a version of the protT5 protein language model on a pfam classification task. This will learn to classify proteins into their respective protein family categories. We'll use the embeddings in this fine-tuned to model to create a vector space. Mapping unknown proteins into this vector space can help predict the function of the unkown protein by judging its similarity to other protein families in the vector space.
