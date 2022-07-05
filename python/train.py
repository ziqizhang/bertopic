from bertopic import BERTopic
import logging, datetime, sys, traceback, pickle
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


file = open("topics.pickle",'rb')
object_file = pickle.load(file)
file = open("topic_probs.pickle",'rb')
object_file = pickle.load(file)

small=[]
for i in range(0, 2000):
    if i%2==0:
        small.append("this is a test sentence for topic modelling")
    else:
        small.append("today is tuesday.")

print("corpus={}".format(len(small)))
topic_model = BERTopic(calculate_probabilities=True)
print("training topics")
topics, probs = topic_model.fit_transform(small)
print("saving topics")
topic_model.save("trained_2.topics")
with open("topics.pickle", 'wb') as outp:
    pickle.dump(topics, outp, pickle.HIGHEST_PROTOCOL)
with open("topic_probs.pickle", 'wb') as outp:
    pickle.dump(probs, outp, pickle.HIGHEST_PROTOCOL)


