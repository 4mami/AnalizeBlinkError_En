import json

file_path = '../Dataset/unseen_mentions/test.json'
out = []
with open(file_path, "r") as file:
    for line in file:
        data: dict = json.loads(line)
        out.append(data)
print("finished")