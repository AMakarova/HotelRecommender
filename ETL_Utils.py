import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
import tensorflow_hub as hub
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")



def get_descriptions(filenames):
    """
    Feed in target embeddings and a data frame containing embeddings for all hotels
    Returns cosine similarities per hotel
    """

    hotels = []
    
    for filename in filenames:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        for i in range(len(root)):
            if root[i].tag == 'COUNTRY':
                for j in range(len(root[i])):
                    if root[i][j].tag == 'DESTINATION':
                        for k in range(len(root[i][j])):
                            if root[i][j][k].tag == 'RESORT':
                                for l in range(len(root[i][j][k])):
                                    if root[i][j][k][l].tag == 'HOTEL':
                                        hotel = []
                                        for m in range(len(root[i][j][k][l])):
                                            if root[i][j][k][l][m].tag == 'ACCOM_UNIT_CODE':
                                                hotel.append(root[i][j][k][l][m].text)
                                            if root[i][j][k][l][m].tag == 'ACCOM_UNIT_NAME':
                                                hotel.append(root[i][j][k][l][m].text)
                                            if root[i][j][k][l][m].tag == 'FEATURED_TEXT':
                                                flag = 0
                                                for n in range(len(root[i][j][k][l][m])):
                                                    if root[i][j][k][l][m][n].tag == 'FEATURE_HEADER':
                                                        if root[i][j][k][l][m][n].text == 'Introduction':
                                                            flag = 1
                                                if flag == 1:
                                                    for n in range(len(root[i][j][k][l][m])):
                                                        if root[i][j][k][l][m][n].tag == 'TEXT':
                                                            hotel.append(root[i][j][k][l][m][n].text)           
                                        hotels.append(hotel)            

    hotels = pd.DataFrame(hotels, columns=['HotelID', 'HotelName', 'HotelDescription'])
    hotels['HotelDescription'] = hotels['HotelDescription'].str.replace('\n', '')
    [hotels] = drop_nas([hotels])  
    hotels = hotels.drop_duplicates()    
    return hotels



def get_embeddings(text):
    """
    Feed in a vector of texts
    Returns a data frame of embeddings
    """
    
    text = text.values.tolist()
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(text))
    return pd.DataFrame(embeddings)



def merge_metadata(metadata, descriptions):
    """
    Feed in structured metadata and hotel descriptions
    Returns a merged dataframe
    """
    
    metadata = pd.merge(metadata, descriptions[['HotelID', 'HotelDescription']], on=['HotelID'], how = 'left')
    metadata = metadata[metadata['HotelDescription'].isnull() == False].reset_index()
    metadata = pd.concat([metadata, get_embeddings(metadata['HotelDescription'])], axis=1, sort=False)
    del metadata['HotelDescription']
    return metadata



def one_hot_encoding(df, columns):
    """
    Feed in the data frame to be transformed as well as column names
    Replaces categorical data with one-hot encodings
    """
    
    import pandas as pd
    
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype('category')
        dummified = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummified], axis = 1)
        del df[col]
        del dummified
    return df



def normalise(df, columns):
    """
    Feed in the data frame to be transformed as well as column names
    Normalises numerical inputs
    """
    
    df = df.copy()
    for col in columns:
        df[col] = (df[col]-df[col].mean())/df[col].std()
    return df



def merge_meta(hotel, cooccurrence, metadata):
    """
    Feed in hotel column name (A or B) and dataframes containing co-purchasing metric and hotel features
    Returns joined dataset (left or right)
    """
    
    X = pd.merge(pd.DataFrame(cooccurrence['Hotel_'+hotel].rename('HotelID')), 
                  metadata, on=['HotelID'], how = 'left')
    return X



def drop_nas(dfs):
    """
    Feed in a list of dataframes
    Drops all rowas in all dataframes that contain missing data
    """
    
    na = pd.isnull(pd.concat(dfs, axis=1, sort=False)).max(axis=1)
    output = []
    for df in dfs:
        df = df[na==False].reset_index(drop=True)
        output.append(df)
    return output



def etl(metadata, cooccurrence, categorical_cols, numerical_cols, outcome_col = 'Cosine_Similarity'):
    """
    Feed in datasets and column names
    Transforms datasets and returns X and Y data frames to be used for modelling
    """

    metadata_norm = one_hot_encoding(metadata, categorical_cols)
    metadata_norm = normalise(metadata_norm, numerical_cols)
    X_left = merge_meta('A', cooccurrence, metadata_norm)
    X_right = merge_meta('B', cooccurrence, metadata_norm)
    Y = pd.DataFrame(cooccurrence[outcome_col])
    X_left, X_right, Y = drop_nas([X_left, X_right, Y])
    return X_left, X_right, Y