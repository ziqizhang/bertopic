'''
For documentation, see https://github.com/MaartenGr/BERTopic
'''
import datetime
import os

from bertopic import BERTopic
import sys

if __name__ == '__main__':
    in_folder=sys.argv[1]
    out_folder=sys.argv[2]
    files= os.listdir(in_folder)
    files.sort()

    for f in files:
        if f.endswith(".topics"):
            print(">>>\tLoading={}, \t{}".format(f, datetime.datetime.now()))
            topic_model=BERTopic.load(in_folder+"/"+f)
            print(">>>\tSaving, \t{}".format(datetime.datetime.now()))
            topic_model.save(out_folder+"/"+f, save_embedding_model=False)
