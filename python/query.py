'''
For documentation, see https://github.com/MaartenGr/BERTopic
'''
import csv
from bertopic import BERTopic
import gensim.corpora as corpora
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel

def save_topic_keywords(topicmodel, outfile, topics=20):
    with open(outfile, 'w', newline='\n') as csvfile:
        header=["topic_id","keyword1","keyword2","keyword3","keyword4","keyword5",
                "keyword6","keyword7","keyword8","keyword9","keyword10"]
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(header)

        count=1
        for t in topicmodel.get_topics():
            if t==-1:
                continue
            words=topic_model.get_topic(t)

            count_words=1
            row=[str(t)]
            for w in words:
                row.append(w[0])
                count_words+=1
                if count_words>10:
                    break
            csvwriter.writerow(row)
            count += 1
            if count>topics:
                break

def calculate_topic_coherence(topic_model, doc_topics_prob, inputdocs):
    # Preprocess Documents
    documents = pd.DataFrame({"Document": inputdocs,
                              "ID": range(len(inputdocs)),
                              "Topic": doc_topics_prob})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                   for topic in range(len(set(doc_topics_prob)) - 1)]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence


if __name__ == '__main__':
    #load a trained topic model
    path='/home/zz/Cloud/GDrive/ziqizhang/hpc/jade/bertopics_min5_cpu/All_Beauty_5.txt.topics'
    topic_model=BERTopic.load(path)

    #for a list of supported queries, see https://github.com/MaartenGr/BERTopic end of page
    #some examples below
    # save_topic_keywords(topic_model,"/home/zz/Cloud/GDrive/ziqizhang/hpc/jade/bertopics_min5_cpu/All_Beauty_5.txt",
    #                     20)

    file1 = open('/home/zz/Cloud/GDrive/ziqizhang/hpc/jade/All_Beauty_5.txt', 'r')
    lines = file1.readlines()
    calculate_topic_coherence(topic_model,10, lines)

    print("end")
