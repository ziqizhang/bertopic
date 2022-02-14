from bertopic import BERTopic
import logging, datetime, sys, traceback
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

small=[]
for i in range(0, 2000):
    small.append("this is a test sentence for topic modelling")

print("corpus={}".format(len(small)))
topic_model = BERTopic()
print("training topics")
topics, probs = topic_model.fit_transform(small)
print("saving topics")
topic_model.save("trained_2.topics")


