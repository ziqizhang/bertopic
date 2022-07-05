import sys, pysolr, csv
from pathlib import Path

'''
code to read Wasim Ahmed's data from a single folder and merge them into a single txt.
input:  /home/zz/Work/data/wasimAHMED 
total=36504
'''

if __name__ == '__main__':

    in_folder=sys.argv[1]
    out_file = sys.argv[2]
    solr_url=sys.argv[3]
    csvfile= open(out_file+'.csv', 'w', newline='', encoding='utf-8')
    csvwriter = csv.writer(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
    csvwriter.writerow(["ID", "Blog", "Post","Text"])

    solr = pysolr.Solr(solr_url, always_commit=True)
    # Do a health check.
    solr.ping()

    with open(out_file, 'w', encoding='utf-8') as file:
        index=0
        batch=[]
        raw_data=sorted(Path(in_folder).rglob('*.txt'))

        for path in raw_data:
            text_file = open(path, "r")
            data = text_file.read().replace("\n"," ").strip()
            text_file.close()
            filename=str(path)
            filename = filename[filename.index(in_folder)+len(in_folder)+1:]
            parts=filename.split("/")

            file.write(data+"\n")
            batch.append({"id":index, "name":filename, "text":data,
                          "blog_s":parts[0], "post_s":parts[1]})#optional
            index += 1

            csvwriter.writerow([index, parts[0], parts[1], data])
            if len(batch)==500:
                print("adding batch (index={})...".format(index))
                solr.add(batch)
                batch.clear()

        if len(batch)>0:
            print("adding final batch...")
            solr.add(batch)
            batch.clear()

        print("total={}".format(index))

    csvfile.close()