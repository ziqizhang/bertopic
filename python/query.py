'''
For documentation, see https://github.com/MaartenGr/BERTopic
'''

from bertopic import BERTopic
import torch

#load a trained topic model
path='/home/zz/Cloud/GDrive/ziqizhang/hpc/jade/bertopics_min5/All_Beauty_5.txt.topics'
topic_model=BERTopic.load(path)

#get basic topic info
print(topic_model.get_topic_info())
