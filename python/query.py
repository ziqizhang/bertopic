'''
For documentation, see https://github.com/MaartenGr/BERTopic
'''

from bertopic import BERTopic
import torch

#load a trained topic model
path='/home/zz/Cloud/GDrive/ziqizhang/hpc/jade/bertopics_min5_cpu/All_Beauty_5.txt.topics'
topic_model=BERTopic.load(path)

#for a list of supported queries, see https://github.com/MaartenGr/BERTopic end of page
#some examples below

#get basic topic info
print(topic_model.get_topic_info())

#go through every topic, print its size
total=0
for t in topic_model.get_topics():
    total+=1
    print(topic_model.get_topic_freq(t))


#get words for a specific topic
print(topic_model.get_topic(0))

#search for topics relevant to a word
#WARNING: only possible on GPU enabled models. If you are using the saved cpu versions,
#this will not work
print(topic_model.find_topics("eyes"))

print("end")
