import gzip
import json

file_path = "./cinbank/dev.jsonl.gz"

def show():
    with gzip.open(file_path, "rt") as f:  # "rt" mode for reading text from gzip
        cnt = 0
        data_list = []
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    
    with open('test.json', 'w') as f:
        json.dump(data_list, f, indent=4)

def change():
    with gzip.open(file_path, "rt") as f:  # "rt" mode for reading text from gzip
        data_list = []
        for line in f:
            data = json.loads(line)

            data['source_texts'] = ['a'] * len(data['source_boxes'])
            data['target_texts'] = data['source_texts']
            data['target_index'] = [x + 1 for x in data['target_index']]
            data_list.append(data)  

    with gzip.open(file_path, "wt") as f:  # "wt" mode for writing text to gzip
        for data in data_list:
            f.write(json.dumps(data) + "\n") 

#change()
show()