import pickle, sys, pysolr
#this file is used after the tm is trained, to add topic ids to each document into the solr index created by preprocess.py

if __name__ == '__main__':
    solr_url = sys.argv[3]
    saved_topics = sys.argv[1]
    saved_topics_prob = sys.argv[2]
    topic_prob_threshold = 0

    file = open(saved_topics, 'rb')
    topics = pickle.load(file)
    file.close()
    file = open(saved_topics_prob, 'rb')
    topics_prob = pickle.load(file)
    file.close()

    solr = pysolr.Solr(solr_url, always_commit=True)
    # search
    stop = False
    start = 0
    while not stop:
        results = solr.search('*:*', **{
            'start': start,
            'rows': 1000,
        })
        if len(results) < 1000:
            stop = True

        print("updating batch, start={}".format(start))
        start += 1000
        # go through every result
        batch_update=[]
        for result in results:
            id = result['id']
            # read the topic from prob matrix
            topics_vector = topics_prob[int(id)]
            topics_id = [x for x in range(0, len(topics_vector))]
            sorted_topics = [x for _, x in sorted(zip(topics_vector, topics_id), reverse=True)]
            result["topic_1st"]=""
            result["topic_2nd"]=""
            result["topic_3rd"]=""
            result["topic_4th"]=""
            result["topic_5th"]=""
            for ordered_topic in range(0, 5):
                if len(topics_id)<= ordered_topic:
                    break
                topic_index = sorted_topics[ordered_topic]
                prob = topics_vector[topic_index]
                if prob > topic_prob_threshold:
                    if ordered_topic == 0:
                        result["topic_1st"] = str(topic_index)
                    elif ordered_topic == 1:
                        result["topic_2nd"] = str(topic_index)
                    elif ordered_topic == 2:
                        result["topic_3rd"] = str(topic_index)
                    elif ordered_topic == 3:
                        result["topic_4th"] = str(topic_index)
                    elif ordered_topic == 4:
                        result["topic_5th"] = str(topic_index)
                else:
                    break
            #append
            batch_update.append(result)
        # update
        solr.add(batch_update, overwrite=True)
    print("completed")