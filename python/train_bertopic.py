# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from bertopic import BERTopic
import sys, os, datetime, logging,traceback, pickle, viz, query, random

from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("bertopic")

def train_bertopic_model(in_file, out_file_folder, out_file_name, min_words=5, sample=None):
    file = open(in_file, 'r')
    tc_file=open(out_file_folder+"/topic_coherence.txt",'a')
    lines=[]

    count_total=0
    count_selected=0
    for l in file:
        count_total+=1
        l=l.strip()
        if len(l.split(" "))<min_words:
            continue
        count_selected+=1
        lines.append(l)

    if sample is not None and len(lines)>sample:
        print(">>>\t\t\ttotal lines={}, sampled={}".format(count_total, sample))
        lines = random.sample(lines, sample)

    print(">>>\t\t\ttotal lines={}, selected={}".format(count_total, count_selected))
    try:
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model = BERTopic(vectorizer_model=vectorizer_model, calculate_probabilities=True)
        print(">>>\t\t\ttraining topics {}".format(datetime.datetime.now()))
        topics, probs = topic_model.fit_transform(lines)
        with open(out_file_folder + "/" + out_file_name + ".topics.pickle", 'wb') as outp:
            pickle.dump(topics, outp, pickle.HIGHEST_PROTOCOL)
        with open(out_file_folder + "/" + out_file_name + ".topic_probs.pickle", 'wb') as outp:
            pickle.dump(probs, outp, pickle.HIGHEST_PROTOCOL)

        print(">>>\t\t\tsaving model {}".format(datetime.datetime.now()))
        topic_model.save(out_file_folder+"/"+out_file_name+".topics")

        print(">>>\t\t\tsaving topic keywords {}".format(datetime.datetime.now()))
        query.save_topic_keywords(topic_model,out_file_folder+"/"+out_file_name+'.keywords.csv', topics=20)

        print(">>>\t\t\tcreating and saving visualization - heatmap {}".format(datetime.datetime.now()))
        try:
            fig  = topic_model.visualize_heatmap(top_n_topics=20)
            fig.write_html(out_file_folder+"/"+out_file_name+".heatmap.html")
            # with open(out_file_folder+"/"+out_file_name+".similarities.pickle", 'wb') as outp:
            #     pickle.dump(matrix, outp, pickle.HIGHEST_PROTOCOL)
        except:
            print(">>>\t\t\t\tunable to create heatmap {}".format(datetime.datetime.now()))
            print(traceback.format_exc())

        print(">>>\t\t\tcreating and saving visualization - topic viz {}".format(datetime.datetime.now()))
        try:
            fig = topic_model.visualize_topics(top_n_topics=20)
            fig.write_html(out_file_folder + "/" + out_file_name + ".viztopic.html")
        except:
            print(">>>\t\t\t\tunable to create topic viz {}".format(datetime.datetime.now()))
            print(traceback.format_exc())
        # fig.write_html("path/to/file.html")

        print(">>>\t\t\tcreating and saving visualization - hierarchy {}".format(datetime.datetime.now()))
        try:
            fig = topic_model.visualize_hierarchy(top_n_topics=20)
            fig.write_html(out_file_folder + "/" + out_file_name + ".hierarchy.html")
        except:
            print(">>>\t\t\t\tunable to create hierarchy viz {}".format(datetime.datetime.now()))
            print(traceback.format_exc())

        print(">>>\t\t\tcreating and saving visualization - barchart {}".format(datetime.datetime.now()))
        try:
            fig = topic_model.visualize_barchart(top_n_topics=10)
            fig.write_html(out_file_folder + "/" + out_file_name + ".barchart.html")
        except:
            print(">>>\t\t\t\tunable to create hierarchy viz {}".format(datetime.datetime.now()))
            print(traceback.format_exc())

        #
        print(">>>\t\t\tcreating and saving visualization - topic/doc, doc {}".format(datetime.datetime.now()))
        try:
            viz.visualize_topic_documents(lines,topics, topic_model, out_file_folder + "/" + out_file_name + ".topicdoc",
                                          topn=20)
        except:
            print(">>>\t\t\t\tunable to create topic/doc viz {}".format(datetime.datetime.now()))
            print(traceback.format_exc())

        print(">>>\t\t\tcalculating topic coherence {}".format(datetime.datetime.now()))
        try:
            tc = query.calculate_topic_coherence(topic_model, topics, lines)
            tc_file.write(out_file_name + "," + str(tc) + "\n")
            print("\t\t\tTC for {}, {}".format(out_file_name, tc))
        except Exception:
            print(">>>\t\tcannot calculate topic coherence, moving on")
            print(traceback.format_exc())

    except:
        print(traceback.format_exc())

    file.close()
    tc_file.close()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    in_folder=sys.argv[1]
    out_folder=sys.argv[2]
    dsample=0
    if len(sys.argv)>3:
        dsample=int(sys.argv[3])

    files=os.listdir(in_folder)
    print(">>>\t\tTotal files={}".format(len(files)))
    files.sort()
    print(">>>\t\tBeginning the process")
    for file in files:
        print(">>>NEW FILE\t\t{} training for {}".format(datetime.datetime.now(), file))
        if dsample>0:
            train_bertopic_model(in_folder+"/"+file, out_folder, file, sample=dsample)
        else:
            train_bertopic_model(in_folder + "/" + file, out_folder, file)
        print(">>>\t\tcompleted")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
