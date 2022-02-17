#see https://github.com/MaartenGr/BERTopic/issues/126

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

import matplotlib, datetime
import matplotlib.pyplot as plt

#%matplotlib inline

def visualize_topic_documents(docs:list, topics, topic_model, outfile):
    embeddings = topic_model._extract_embeddings(docs, method="document")
    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings)
    df = pd.DataFrame(umap_model.embedding_, columns=["x", "y"])
    df["topic"] = topics

    # Plot parameters
    top_n = 10
    fontsize = 15

    to_plot = df.copy()
    to_plot[df.topic >= top_n] = -1
    outliers = to_plot.loc[to_plot.topic == -1]
    non_outliers = to_plot.loc[to_plot.topic != -1]

    cmap = matplotlib.colors.ListedColormap(['#FF5722', # Red
                                             '#03A9F4', # Blue
                                             '#4CAF50', # Green
                                             '#80CBC4', # FFEB3B
                                             '#673AB7', # Purple
                                             '#795548', # Brown
                                             '#E91E63', # Pink
                                             '#212121', # Black
                                             '#00BCD4', # Light Blue
                                             '#CDDC39', # Yellow/Red
                                             '#AED581', # Light Green
                                             '#FFE082', # Light Orange
                                             '#BCAAA4', # Light Brown
                                             '#B39DDB', # Light Purple
                                             '#F48FB1', # Light Pink
                                             ])

    fig, ax = plt.subplots(figsize=(15, 15))
    scatter_outliers = ax.scatter(outliers['x'], outliers['y'], c="#E0E0E0", s=1, alpha=.3)
    scatter = ax.scatter(non_outliers['x'], non_outliers['y'], c=non_outliers['topic'], s=1, alpha=.3, cmap=cmap)

    ax.text(0.99, 0.01, f"BERTopic - Top {top_n} topics", transform=ax.transAxes, horizontalalignment="right", color="black")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(outfile+"_1.pdf", format='pdf')
    plt.clf()
    #plt.show()

    fig2, ax2 = plt.subplots(figsize=(15, 15))
    scatter_outliers = ax2.scatter(outliers['x'], outliers['y'], c="#E0E0E0", s=1, alpha=.3)
    scatter = ax2.scatter(non_outliers['x'], non_outliers['y'], c=non_outliers['topic'], s=1, alpha=.3, cmap=cmap)

    ax2.text(0.99, 0.01, f"BERTopic - Top {top_n} topics", transform=ax.transAxes, horizontalalignment="right",
            color="black")
    plt.xticks([], [])
    plt.yticks([], [])
    #print("{} Add topic names to clusters".format(datetime.datetime.now()))
    centroids = to_plot.groupby("topic").mean().reset_index().iloc[1:]
    for row in centroids.iterrows():
        topic = int(row[1].topic)
        text = f"{topic}: " + "_".join([x[0] for x in topic_model.get_topic(topic)[:5]])
        ax2.text(row[1].x, row[1].y * 1.01, text, fontsize=fontsize, horizontalalignment='center')
    plt.savefig(outfile+"_2.pdf", format='pdf')
    #plt.show()

if __name__ == '__main__':
    print("{} Create topic model".format(datetime.datetime.now()))
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    small = docs[0:3000]
    print("corpus={}".format(len(small)))
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    print("training topics")
    topics, probs = topic_model.fit_transform(small)
    visualize_topic_documents(small, topics, topic_model,"")