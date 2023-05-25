import json
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re
from tqdm import tqdm
import Twitter.keywords as t

def tweets_with_position_amount(d):
    """How many tweets have geographical information available"""
    c = 0
    for user in d.keys():
        if d[user]:
            for tw in d[user].keys():
                if "geo" in d[user][tw].keys():
                    c += 1
    return c

def tweets_per_keyword(d):
    d_stats = {}
    for w in tqdm(t.word_list):
        c = 0
        for user in d.keys():
            if d[user]:
                for tw in d[user].keys():
                    txt = d[user][tw]['text']
                    if txt.lower().__contains__(w.lower()):
                        c += 1
        d_stats[w] = c

def plot_dictionary_histogram(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    # Plot the histogram
    plt.bar(keys, values)
    # Set labels and title
    plt.xlabel('Keywords')
    plt.ylabel('No. tweets')
    plt.title('Distribution of the keywords in the tweets')
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90)
    # Display the plot
    plt.show()

"""with open("Twitter/tweets_per_user.json", 'r') as f:
    d = json.load(f)
d_stats = tweets_per_keyword(d)
plot_dictionary_histogram(d_stats)"""

def clean_text(text):
    """
    Apply NLP pipeline to the text. The actions performed are tokenization, punctuation removal, stopwords removal, stemming
    """
    stemmer = PorterStemmer()
    text = text.lower()
    t = re.sub(r'\(\+photos*\)|\(\+videos*\)|\(\+images*\)|\[[^\]]*\]', "", text)   # First remove placeholders
    t = re.sub(r'\w*@\w+|\b(?:https?://)\S+\b|[_"\-;“”%()|+&=~*%’.,!?:#$\[\]/]', "", t)   # Remove tags, links and apostrophe
    t = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDCF\uFDF0-\uFDFF\uFE70-\uFEFF]', "", t)      # Remove arabic characters
    splitted = t.split()
    cleaned = []
    for w in splitted:
        w = w.lower()
        w = stemmer.stem(w)
        cleaned.append(w)
    cleaned = [w for w in cleaned if w not in stopwords.words('english')]
    return " ".join(cleaned)

def clean_dataframe(df: pd.DataFrame, id_column, text_column):
    """Preprocess the textual content of the dataframe, ignore useless columns, rename the columns with text and id """
    new_list = []
    for index, row in tqdm(df.iterrows()):
        dict_row = {}
        if pd.isna(row[text_column]):
            pass
        else:
            dict_row[id_column] = row[id_column]
            dict_row[text_column] = clean_text(row[text_column])
            new_list.append(dict_row)
    cleaned_df = pd.DataFrame(new_list)
    return cleaned_df

def concat(l: pd.Series):
    l = l.tolist()
    return " ".join(l)


def concatenate_posts(df, aggregator, text_column):
    """
    Take a df having one row for each post, return a new df having one row for each user, and as values the concatenation of the posts made by that user
    Args:
        df: Dataframe on which the concatenation will be performed
        aggregator: Name of the column along which the posts will be concatenated (ie the ID column)
        text_column: Name of the column containing the text that will be concatenated

    Returns:

    """
    ser = df.groupby(aggregator)[text_column].apply(concat)
    df = pd.DataFrame(columns=[aggregator, text_column])
    df[aggregator] = ser.index
    df[text_column] = ser.values
    return df


if __name__ == "__main__":
    # Load and preprocess the dataframe containing the risky tweets. The so-obtained dataset will be used for building a
    # w2v vector that will act as reference for assessing whether a user can be labeled as risky or not depending on how
    # close his tweets are to the reference
    d = pd.read_csv("evil/cleaned/no_affiliations.csv")
    # d.drop(d[d.level.values=="0 - Negative"].index, inplace=True)
    d = clean_dataframe(d, id_column="level", text_column="text")
    d.to_csv("evil/cleaned/preprocessed.csv")
