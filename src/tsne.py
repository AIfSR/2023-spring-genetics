from sklearn.manifold import TSNE


def tsne_transform(embeddings, **kwargs):

    tsne_params = dict()

    tsne_params['n_components'] = kwargs.get('n_components', 2) #number of dimenaions in the output space
    tsne_params['perplexity'] = kwargs.get('perplexity', 6) #The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity
    tsne_params['random_state'] = kwargs.get('random_state', 42) #for reproducibality
    tsne_params['n_iter'] = kwargs.get('n_iter', 15000) #number of iterations for optimization
    tsne_params['verbose'] = kwargs.get('verbose', 1) 
    tsne_params['n_jobs'] = kwargs.get('n_jobs', -1)
    tsne_params['metric'] = kwargs.get('metric', 'cosine') #need to check this!

    transformed_embeddings = TSNE(**tsne_params).fit_transform(embeddings)

    return transformed_embeddings

