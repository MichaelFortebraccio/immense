import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from os.path import exists
from keras.models import load_model
from text_preprocessing import TextPreprocessing
from word_embedding import WordEmb
from ae import AE
from graph_embeddings.node2vec import Node2VecEmbedder
from decision_tree import *
from utils import prepare_for_decision_tree


seed = 123
np.random.seed(seed)


if __name__ == "__main__":
    """# train W2V on Twitter dataset
    df_tweets = pd.read_csv('tweet_labeled.csv', sep=',')
    tok = TextPreprocessing()
    tweet = tok.token_list(df_tweets['text_cleaned'].tolist())

    exit()
    w2v_model = WordEmb(tweet)

    # split tweet in safe and dangerous
    dang_tweets = df_tweets.loc[df_tweets['label'] == 1]['text_cleaned']
    safe_tweets = df_tweets.loc[df_tweets['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    w2v_exist = exists('model/w2v_text.h5')
    if not w2v_exist:
        w2v_model.train_w2v()

    # convert text to vector
    list_dang_tweets_exist = exists('textual_data/list_dang_tweets.pickle')
    if list_dang_tweets_exist:
        with open('textual_data/list_dang_tweets.pickle', 'rb') as handle:
            list_dang_tweets = pickle.load(handle)
    else:
        list_dang_tweets = w2v_model.text_to_vec(list_dang_tweets)
        with open('textual_data/list_dang_tweets.pickle', 'wb') as handle:
            pickle.dump(list_dang_tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    list_safe_tweets_exist = exists('textual_data/list_safe_tweets.pickle')
    if list_safe_tweets_exist:
        with open('textual_data/list_safe_tweets.pickle', 'rb') as handle:
            list_safe_tweets = pickle.load(handle)
    else:
        list_safe_tweets = w2v_model.text_to_vec(list_safe_tweets)
        with open('textual_data/list_safe_tweets.pickle', 'wb') as handle:
            pickle.dump(list_safe_tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ###############################

    #################train/load autoencoder safe and dang
    model_safe_exist = exists('model/autoencodersafe.h5')
    model_dang_exist = exists('model/autoencoderdang.h5')

    if model_dang_exist:
        model_dang = load_model('model/autoencoderdang.h5')     # load existing model
    else:
        print("Training dangerous autoencoder")
        AE(input_len=512, X_train=list_dang_tweets, label='dang').train_autoencoder()   # train the model
        model_dang = load_model('model/autoencoderdang.h5')

    if model_safe_exist:
        model_safe = load_model('model/autoencodersafe.h5')  # load existing model
    else:
        print("Training safe autoencoder")
        AE(input_len=512, X_train=list_safe_tweets, label='safe').train_autoencoder()  # train the model
        model_safe = load_model('model/autoencodersafe.h5')
    ##########################
"""
    df = pd.read_csv("tweet_labeled.csv")[['label', 'id']]
    emb_relationships = Node2VecEmbedder(path_to_edges="graph_embeddings/stuff/network.edg", number_of_walks=10, walk_length=10,
                                         embedding_size=128, p=1, q=4, epochs=10, save_path="graph_embeddings/stuff/n2v_128_rel.model")
    emb_closeness = Node2VecEmbedder(path_to_edges="graph_embeddings/stuff/closeness_network.edg", number_of_walks=10, walk_length=10,
                                     embedding_size=128, p=1, q=4, epochs=10, save_path="graph_embeddings/stuff/n2v_128_rel.model")

    if not emb_relationships.load_model():
        print("Learning relationships n2v model")
        emb_relationships.learn_n2v_embeddings()
    mod_rel = emb_relationships.load_model()

    if not emb_closeness.load_model():
        print("Learning closeness relationships n2v model")
        emb_closeness.learn_n2v_embeddings()
    mod_clos = emb_closeness.load_model()

    X_train, X_test, Y_train, Y_test = train_test_split(df['id'].to_list(), df['label'].to_list(), test_size=0.2,
                                                        train_size=0.8)
    dt_relationships = train_decision_tree(train_set_ids=X_train, train_set_labels=Y_train, n2v_model=mod_rel)
    dt_closeness = train_decision_tree(train_set_ids=X_train, train_set_labels=Y_train, n2v_model=mod_clos)


    #test_decision_tree(test_set=X_test, test_set_labels=Y_test, cls=cls_relationships)



