'''
code based on https://gist.github.com/Rapptz/c4324f17a80c94776832430007ad40e6
'''
from sentence_transformers import SentenceTransformer
import numpy as np
import random, umap, sys, os, datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster(in_file, out_file_folder, model_path,min_words=5, sample=None):
    name=in_file[in_file.rindex('/')+1:]
    file = open(in_file, 'r')
    lines = []

    count_total = 0
    count_selected = 0
    for l in file:
        count_total += 1
        l = l.strip()
        if len(l.split(" ")) < min_words:
            continue
        count_selected += 1
        lines.append(l)

    if sample is not None and len(lines) > sample:
        print(">>>\t\t\ttotal lines={}, sampled={}".format(count_total, sample))
        lines = random.sample(lines, sample)

    print(">>>\t\t\ttotal lines={}, selected={}".format(count_total, count_selected))

    model = SentenceTransformer(model_path)
    embeddings = model.encode(lines, show_progress_bar=True)
    # Save embeddings
    all_embeddings = np.array(embeddings)
    np.save(out_file_folder+"/"+name+"_embeddings.npy", all_embeddings)

    if len(lines)<10:
        print(">>>\t\tinsufficient instances to cluster {}".format(len(lines)))
        return
    kmeans = KMeans(n_clusters=10, max_iter=100, algorithm='auto').fit(embeddings)

    outfile=out_file_folder+"/"+name+"_cluster_"
    for n in (3, 5, 15, 30):
        umap_data = umap.UMAP(n_neighbors=n, n_components=2, min_dist=0.25, metric='cosine').fit_transform(embeddings)
        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = kmeans.labels_
        # Visualize clusters
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=2.8)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=3.5, cmap='hsv_r')
        plt.colorbar()
        plt.savefig(outfile + str(n)+".png", format='png', dpi=300)
        plt.clf()
        plt.close()

if __name__ == '__main__':
    in_folder=sys.argv[1]
    out_folder=sys.argv[2]
    model_path=sys.argv[3]
    dsample=0
    if len(sys.argv)>4:
        dsample=int(sys.argv[4])

    files=os.listdir(in_folder)
    print(">>>\t\tTotal files={}".format(len(files)))
    files.sort()
    print(">>>\t\tBeginning the process")
    for file in files:
        print(">>>\t\t{} processing {}".format(datetime.datetime.now(), file))
        if dsample>0:
            cluster(in_folder+"/"+file, out_folder, model_path, sample=dsample)
        else:
            cluster(in_folder + "/" + file, out_folder, model_path)
        print(">>>\t\tcompleted")