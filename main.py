import pandas as pd
import numpy as np
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
from utils import prepare_for_decision_tree, create_or_load_tweets_list


seed = 123
np.random.seed(seed)


if __name__ == "__main__":
    df_tweets = pd.read_csv('tweet_labeled.csv', sep=',')
    '''X_train, X_test, Y_train, Y_test = train_test_split(df_tweets['id'].to_list(), df_tweets['label'].to_list(),
                                                        test_size=0.2, train_size=0.8)'''
    # train W2V on Twitter dataset
    tok = TextPreprocessing()
    tweet = tok.token_list(df_tweets['text_cleaned'].tolist())
    w2v_model = WordEmb(tweet)

    # split tweet in safe and dangerous
    dang_tweets = df_tweets.loc[df_tweets['label'] == 1]['text_cleaned']
    safe_tweets = df_tweets.loc[df_tweets['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    if not exists('model/w2v_text.h5'):
        w2v_model.train_w2v()
    #print(list_dang_tweets)
    # convert text to vector
    list_dang_tweets = create_or_load_tweets_list(path='textual_data/list_dang_tweets.pickle', w2v_model=w2v_model, tokenized_list=list_dang_tweets)
    list_safe_tweets = create_or_load_tweets_list(path='textual_data/list_safe_tweets.pickle', w2v_model=w2v_model, tokenized_list=list_safe_tweets)

    #################TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER####################
    dang_ae = AE(input_len=512, X_train=list_dang_tweets, label='dang').load_autoencoder()
    safe_ae = AE(input_len=512, X_train=list_safe_tweets, label='safe').load_autoencoder()

    list_tweets_embs = w2v_model.text_to_vec(tweet)
    prediction_dang = dang_ae.predict(list_tweets_embs)
    prediction_safe = safe_ae.predict(list_tweets_embs)

    tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(list_tweets_embs, dtype=tf.float32)).numpy()
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, tweets_sigmoid).numpy().tolist()
    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, tweets_sigmoid).numpy().tolist()

    ##########################TRAIN AND LOAD GRAPH EMBEDDING MODELS####################
    rel_tree_path = "model/rel_dtree.h5"
    clos_tree_path = "model/clos_dtree.h5"
    if not exists(rel_tree_path):
        mod_rel = Node2VecEmbedder(path_to_edges="graph_embeddings/stuff/network.edg", weighted=False, directed=True,
                                    n_of_walks=10, walk_length=10, embedding_size=128, p=1, q=4, epochs=100, rel_type="rel").load_model()
        train_decision_tree(train_set_ids=df_tweets['id'], train_set_labels=df_tweets['label'], n2v_model=mod_rel, save_path=rel_tree_path)

    if not exists(clos_tree_path):
        mod_clos = Node2VecEmbedder(path_to_edges="graph_embeddings/stuff/closeness_network.edg", weighted=True, directed=False,
                                    n_of_walks=10, walk_length=10, embedding_size=128, p=1, q=4, epochs=100, rel_type="closeness").load_model()
        train_decision_tree(train_set_ids=df_tweets['id'], train_set_labels=df_tweets['label'], n2v_model=mod_clos, save_path=clos_tree_path)

    tree_rel = load_decision_tree(rel_tree_path)
    tree_clos = load_decision_tree(clos_tree_path)


    #test_decision_tree(test_set=X_test, test_set_labels=Y_test, cls=cls_relationships)
