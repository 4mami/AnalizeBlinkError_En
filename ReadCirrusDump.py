import json
import gzip
import tqdm

file_path = '../Dataset/enwiki-20201123-cirrussearch-content.json.gz'
counter = 0
with gzip.open(file_path) as file:
    with open("output_file.json", "w") as output_file:
        output_file.write("[\n")

        for line in tqdm.tqdm(file):
            data: dict = json.loads(line)
            counter += 1
            if "index" not in data:
                json_str = json.dumps(data, indent=2)
                output_file.write(json_str + ",\n\n\n")
            if (counter > 10000):
                break

        output_file.write("]")