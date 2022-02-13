from bertopic import BERTopic
import logging, datetime
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
# small=docs[0:5000]
# print("corpus={}".format(len(small)))
# topic_model = BERTopic()
# print("training topics")
# topics, probs = topic_model.fit_transform(small)
# print("saving topics")
# topic_model.save("trained_2.topics")

topic_model=BERTopic.load("trained_2.topics")
print("{}\tvisualise".format(datetime.datetime.now()))
fig=topic_model.visualize_topics()
fig.show()

fig=topic_model.visualize_hierarchy()
fig.show()

fig=topic_model.visualize_barchart()
fig.show()

fig=topic_model.visualize_heatmap()
fig.show()
#
fig=topic_model.visualize_term_rank()
fig.show()

print("done")
