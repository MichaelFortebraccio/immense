import json
import nltk
import pandas as pd
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from tqdm import tqdm


def tweets_with_position_amount(d):
    """How many tweets have geographical information available"""
    c = 0
    for user in d.keys():
        if d[user]:
            for tw in d[user].keys():
                if "geo" in d[user][tw].keys():
                    c += 1
    return c


def clean_text(text):
    """
    Apply NLP pipeline to the text. The actions performed are tokenization, punctuation removal, stopwords removal, stemming
    """
    #stemmer = PorterStemmer()
    text = text.lower()
    t = re.sub(r'\(\+photos*\)|\(\+videos*\)|\(\+images*\)|\[[^\]]*\]|^rt', "", text)   # First remove placeholders
    t = re.sub(r'\w*@\w+|\b(?:https?://)\S+\b|[_"\-;“”%()|+&=~*%’.,!?:#$\[\]/]', "", t)   # Remove tags, links and apostrophe
    t = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDCF\uFDF0-\uFDFF\uFE70-\uFEFF]', "", t)      # Remove arabic characters
    splitted = t.split()
    cleaned = []
    for w in splitted:
        w = w.lower()
        #w = stemmer.stem(w)
        cleaned.append(w)
    cleaned = [w for w in cleaned if w not in stopwords.words('english')]
    return " ".join(cleaned)


def clean_dataframe(df: pd.DataFrame, text_column):
    """
    Preprocess the textual content of the dataframe, ignore useless columns, rename the columns with text and id
    :param df: Dataframe
    :param text_column: Name of the column containing the text to preprocess
    :return: dataframe cleaned
    """
    new_list = []
    non_text_columns = [c for c in df.columns if c != text_column]
    for index, row in tqdm(df.iterrows()):
        dict_row = {}
        if pd.isna(row[text_column]):
            pass
        else:
            for c in non_text_columns:
                dict_row[c] = row[c]
            dict_row[text_column] = clean_text(row[text_column])
            new_list.append(dict_row)
    cleaned_df = pd.DataFrame(new_list)
    return cleaned_df


def concat(l: pd.Series):
    l = l.tolist()
    return " ".join(l)


def concatenate_posts(df, aggregator_column, text_column):
    """
    Take a df having one row for each post, return a new df having one row for each user, and as values the concatenation of the posts made by that user
    Args:
        df: Dataframe on which the concatenation will be performed
        aggregator_column: Name of the column along which the posts will be concatenated (ie the ID column)
        text_column: Name of the column containing the text that will be concatenated

    Returns:
    """
    ser = df.groupby(aggregator_column)[text_column].apply(concat)
    df = pd.DataFrame(columns=[aggregator_column, text_column])
    df[aggregator_column] = ser.index
    df[text_column] = ser.values
    return df
