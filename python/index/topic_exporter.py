#this reads the solr index of the data and their topics (created by topic_indexer) and output one csv
#per topic, showing
#doc id, blog identifier, text, topic id, and topic ranking (1st, 2nd, 3rd only)
import sys,pysolr,csv

if __name__ == '__main__':
    solr_url = sys.argv[1]
    topN_topics=int(sys.argv[2])
    outfolder=sys.argv[3]
    solr = pysolr.Solr(solr_url, always_commit=True)

    #for top N topics
    for topic_id in range(0, topN_topics):
        print("topic={}".format(topic_id))
        csvfile = open(outfolder + '/topic_{}.csv'.format(topic_id), 'w', newline='', encoding='utf-8')
        csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(["ID", "Blog", "Post", "Topic ID","Topic Ranking", "Text"])
        # search
        stop = False
        start = 0
        while not stop:
            results = solr.search('topic_1st:{} OR topic_2nd:{} OR topic_3rd:{}'.format(topic_id, topic_id, topic_id), **{
                'start': start,
                'rows': 1000,
            })
            if len(results) < 1000:
                stop = True

            print("\tprocessing batch, start={}".format(start))
            start += 1000
            # go through every result
            batch_update = []
            for result in results:
                id = result['id']
                # read the topic from prob matrix
                blog=result['blog_s']
                post=result['post_s']
                text=result['text']
                if result['topic_1st']==str(topic_id):
                    ranking='1'
                elif result['topic_2nd']==str(topic_id):
                    ranking='2'
                elif result['topic_3rd']==str(topic_id):
                    ranking='3'
                else:
                    ranking='other'
                rec=[id, blog,post,topic_id,ranking, text]
                csvwriter.writerow(rec)
        csvfile.close()
    print("completed")