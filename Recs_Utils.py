import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns



def compare_embeddings(target_hotel, hotel_emb, cooccurrence=None):
    """
    Feed in target embeddings and a data frame containing embeddings for all hotels
    Returns cosine similarities per hotel
    """
    
    cosine_sim = []
    target_emb = hotel_emb[hotel_emb['HotelID'] == target_hotel].iloc[:,1:]

    for h in range(hotel_emb.shape[0]):
        cosine_sim.append(cosine_similarity(target_emb.values.reshape(1, -1),
                                           hotel_emb.iloc[h,1:].values.reshape(1, -1))[0][0]) 
    cosine_sim = pd.concat([hotel_emb.iloc[:,0], pd.DataFrame(cosine_sim, columns = ['Modelled_Cosine_Similarity'])], axis=1)
    
    if cooccurrence is not None:
        cooccur = pd.DataFrame(cooccurrence[['Hotel_B', 'Cosine_Similarity']][cooccurrence['Hotel_A']==target_hotel].values, 
                            columns = ['HotelID', 'Cosine_Similarity'])
        cosine_sim = pd.merge(cosine_sim, cooccur, on=['HotelID'], how = 'left')
    
    return cosine_sim



def closest_hotels(metadata, cosine_sim, n=10):
    """
    Feed in a list of hotels and their similarity scores
    Returns top n hotels that are the most similar to target hotel
    """
    
    merge = pd.merge(metadata, cosine_sim, on=['HotelID'], how = 'left')
    
    return merge.sort_values('Modelled_Cosine_Similarity', ascending = False).head(n)



def get_tsne(hotel_emb, metadata):
    """
    Feed in hotel embeddings and metadata
    Returns a dataframe of 2D TSNE values joined with original metadata
    """
    
    tsne = pd.DataFrame(TSNE(2, metric='cosine').fit_transform(hotel_emb.iloc[:, 1:]), 
                        columns=['TSNE 1', 'TSNE 2'])
    tsne = pd.concat([hotel_emb.iloc[:,0], tsne], axis=1)
    tsne = pd.merge(tsne, metadata, on=['HotelID'], how = 'left')
    return tsne



def plot_tsne(tsne, column, categorical):
    """
    Feed in TSNE dataset and one of the predictor features
    Returns a plot of TSNE values coloured by predictor feature
    """

    if column in categorical:
        palette = 'tab20'
        tsne[column] = tsne[column].astype('category')
    else:
        palette = 'RdYlGn'

    lm = sns.lmplot(x='TSNE 1', y='TSNE 2', data=tsne, fit_reg=False, hue=column, 
               legend=False, height=7, aspect=3/2, palette=palette)

    lm.fig.suptitle("TSNE coloured by "+column, y=1, fontsize=24)
    return lm