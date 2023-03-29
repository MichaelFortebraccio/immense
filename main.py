import pandas as pd
import numpy as np
from os.path import exists
from modelling.sairus import train
import gdown
seed = 123
np.random.seed(seed)

#if __name__ == "__main__":
def main(textual_content_link, social_graph, closeness_graph, word_embedding_size=512, window=5, w2v_epochs=10,
         spat_node_embedding_size=128, rel_node_embedding_size=128, n_of_walks_spat=10, n_of_walks_rel=10,
         walk_length_spat=10, walk_length_rel=10, p_spat=1, p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100):
    posts_path = "dataset/posts_labeled.csv"
    social_path = "dataset/social_network.edg"
    closeness_path = "dataset/closeness_network.edg"
    if not exists(posts_path):
        gdown.download(url=textual_content_link, output=posts_path, quiet=False, fuzzy=True)
    if not exists(social_path):
        gdown.download(url=social_graph, output=social_path, quiet=False, fuzzy=True)
    if not exists(closeness_path):
        gdown.download(url=closeness_graph, output=closeness_path, quiet=False, fuzzy=True)

    train_path = "dataset/train.csv"
    test_path = "dataset/test.csv"
    df = pd.read_csv('dataset/posts_labeled.csv', sep=',')
    if not exists(train_path) or not exists(test_path):
        df = df.sample(frac=1, random_state=1).reset_index()    # Shuffle the dataframe
        idx = round(len(df)*0.8)
        train_df = df[:idx]
        test_df = df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    
    # dang_ae, safe_ae, w2v_model, n2v_rel, n2v_clos, tree_rel, tree_clos, mlp = 
    train(train_df=train_df, word_embedding_size=word_embedding_size, window=window, w2v_epochs=w2v_epochs, 
          spat_node_embedding_size=spat_node_embedding_size, rel_node_embedding_size=rel_node_embedding_size,
          n_of_walks_spat=n_of_walks_spat, n_of_walks_rel=n_of_walks_rel, walk_length_spat=walk_length_spat,
          walk_length_rel=walk_length_rel, p_spat=p_spat, p_rel=p_rel, q_spat=q_spat, q_rel=q_rel,
          n2v_epochs_spat=n2v_epochs_spat, n2v_epochs_rel=n2v_epochs_rel, path_to_edges_rel=social_path,
          path_to_edges_spat=closeness_path, model_dir="models", dataset_dir="dataset")
    # test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_clos=tree_clos, n2v_rel=n2v_rel, n2v_clos=n2v_clos, mlp=mlp)





    '''# train W2V on Twitter dataset
    tok = TextPreprocessing()
    tweet = tok.token_list(train_df['text_cleaned'].tolist())
    w2v_model = WordEmb(tweet)

    # split tweet in safe and dangerous
    dang_tweets = train_df.loc[train_df['label'] == 1]['text_cleaned']
    safe_tweets = train_df.loc[train_df['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    if not exists('models/w2v_text.h5'):
        w2v_model.train_w2v()

    # convert text to vector
    list_dang_tweets = create_or_load_tweets_list(path='dataset/textual_data/list_dang_tweets.pickle', w2v_model=w2v_model, tokenized_list=list_dang_tweets)
    list_safe_tweets = create_or_load_tweets_list(path='dataset/textual_data/list_safe_tweets.pickle', w2v_model=w2v_model, tokenized_list=list_safe_tweets)

    #################TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER####################
    dang_ae = AE(input_len=512, X_train=list_dang_tweets, label='dang').load_autoencoder()
    safe_ae = AE(input_len=512, X_train=list_safe_tweets, label='safe').load_autoencoder()

    ################# TRAIN OR LOAD DECISION TREES ####################
    rel_tree_path = "models/dtree_rel.h5"
    clos_tree_path = "models/dtree_clos.h5"
    path_to_edges_rel = "graph_embeddings/stuff/network.edg"
    path_to_edges_clos = "graph_embeddings/stuff/closeness_network.edg"
    rel_n2v_path = "models/n2v_rel.h5"
    clos_n2v_path = "models/n2v_clos.h5"

    n2v_rel = Node2VecEmbedder(path_to_edges=path_to_edges_rel, weighted=False, directed=True, n_of_walks=10,
                               walk_length=10, embedding_size=128, p=1, q=4, epochs=100, model_path=rel_n2v_path,
                               name="relationships").learn_n2v_embeddings()
    #.load_model()
    if not exists(rel_tree_path):   # IF THE DECISION TREE HAS NOT BEEN LEARNED, LOAD/TRAIN THE N2V MODEL
        train_set_ids_rel = [i for i in train_df['id'] if str(i) in n2v_rel.wv.key_to_index]
        train_decision_tree(train_set_ids=train_set_ids_rel, save_path=rel_tree_path, n2v_model=n2v_rel,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_rel)]['label'], name="relationships")

    n2v_clos = Node2VecEmbedder(path_to_edges=path_to_edges_clos, weighted=True, directed=False, n_of_walks=10,
                                walk_length=10, embedding_size=128, p=1, q=4, epochs=100, model_path=clos_n2v_path,
                                name="closeness").learn_n2v_embeddings()
        #load_model()
    if not exists(clos_tree_path):
        train_set_ids_clos = [i for i in train_df['id'] if str(i) in n2v_clos.wv.key_to_index]
        train_decision_tree(train_set_ids=train_set_ids_clos, save_path=clos_tree_path, n2v_model=n2v_clos,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_clos)]['label'], name="closeness")

    tree_rel = load_decision_tree(rel_tree_path)
    tree_clos = load_decision_tree(clos_tree_path)
    difference_rel = difference_clos = []

    ################# NOW THAT WE HAVE THE MODELS WE CAN OBTAIN THE TRAINING SET FOR THE MLP #################
    dataset = np.zeros((len(train_df), 7))

    list_tweets_embs = w2v_model.text_to_vec(tweet)
    prediction_dang = dang_ae.predict(list_tweets_embs)
    prediction_safe = safe_ae.predict(list_tweets_embs)

    tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(list_tweets_embs, dtype=tf.float32)).numpy()
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, tweets_sigmoid).numpy()
    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, tweets_sigmoid).numpy()
    labels = [1 if i<j else 0 for i,j in zip(prediction_loss_dang, prediction_loss_safe)]
    dataset[:, 0] = prediction_loss_dang
    dataset[:, 1] = prediction_loss_safe
    dataset[:, 2] = np.array(labels)
    for index, row in tqdm(train_df.iterrows()):
        id = str(row['id'])
        if id in n2v_rel.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_rel, n2v_model=n2v_rel)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 3] = pr
        dataset[index, 4] = conf
        if id in n2v_clos.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_clos, n2v_model=n2v_clos)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 5] = pr
        dataset[index, 6] = conf

    mlp = MLP(X_train=dataset, y_train=np.array(train_df['label']))
    mlp.train()

    ######## TESTING ########
    test_set = np.zeros(shape=(len(test_df), 7))
    tweet_test = tok.token_list(test_df['text_cleaned'].tolist())

    test_tweets_embs = w2v_model.text_to_vec(tweet_test)
    p_dang = dang_ae.predict(test_tweets_embs)
    p_safe = safe_ae.predict(test_tweets_embs)

    test_tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(test_tweets_embs, dtype=tf.float32)).numpy()
    p_loss_dang = tf.keras.losses.mse(p_dang, test_tweets_sigmoid).numpy()
    p_loss_safe = tf.keras.losses.mse(p_safe, test_tweets_sigmoid).numpy().tolist()
    test_labels = [1 if i < j else 0 for i, j in zip(p_loss_dang, p_loss_safe)]
    test_set[:, 0] = p_loss_dang
    test_set[:, 1] = p_loss_safe
    test_set[:, 2] = np.array(test_labels)

    # At test time, if we meet an instance that doesn't have information about relationships or closeness, we will
    # replace the decision tree prediction with the most frequent label in the training set
    pred_missing_info = train_df['label'].value_counts().argmax()
    conf_missing_info = max(train_df['label'].value_counts())/len(train_df)  # ratio
    a = 0
    for index, row in tqdm(test_df.iterrows()):
        id = row['id']
        if str(id) in n2v_rel.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_rel,
                                          n2v_model=n2v_rel)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 3] = pr
        test_set[index, 4] = conf
        if str(id) in n2v_clos.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_clos,
                                          n2v_model=n2v_clos)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 5] = pr
        test_set[index, 6] = conf

    mlp.test(test_set, np.array(test_df['label']))'''
