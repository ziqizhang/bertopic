import pickle, sys

if __name__ == '__main__':
    solr_url=sys.argv[3]
    saved_topics=sys.argv[1]
    saved_topics_prob=sys.argv[2]

    file = open(saved_topics,'rb')
    topics= pickle.load(file)
    file.close()
    file = open(saved_topics_prob,'rb')
    topics_prob=pickle.load(file)
    file.close()