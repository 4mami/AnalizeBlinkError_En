import re
import copy
import gzip
import json
import sys
import tqdm
from logging import getLogger, config

CIRRUSSEARCH_FILE_PATH = "../Dataset/enwiki-20201123-cirrussearch-content.json.gz"
HT_FILE_PATH = "outputs/query_result.txt"
OUTPUT_FILE_PATH = "outputs/preprocessed_dumps.json"
PATTERN = r' in | from | for | of | by | for | involving '

class PreProcessedCirrusDump:
    title = ""
    heading = [] # 元のcirrus dumpのデータすべてにある
    category = [] # 元のcirrus dumpのデータすべてにある
    roughen_category = []
    ores_articletopics = [] # optional
    ores_articletopic = [] # optional

    def __init__(self, ht_categories, title, heading, category, ores_articletopics=None, ores_articletopic=None):
        self.title = title
        self.heading = heading
        if (ores_articletopic is not None):
            self.ores_articletopic = ores_articletopic
        if (ores_articletopics is not None):
            self.ores_articletopics = ores_articletopics
        self.category = copy.copy(category)
        # 隠しカテゴリ・追跡カテゴリを除去する
        lowered_categories = list(map(lambda c: c.lower(), category))
        categories_without_ht = [c for c in lowered_categories if c not in ht_categories]

        self.roughen_category = self.RoughenCategory(categories_without_ht)

    def RoughenCategory(self, old_categories: list): # 元のカテゴリ配列の要素を書き換えても特に問題はないので、普通に参照渡しで
        new_categories = []
        for category in old_categories:
            # 元のカテゴリを、前置詞（‘in’, ‘from’, ‘for’, ‘of’, ‘by’, ‘for’, ‘involving'）で2つに分割する「car of united states for employee」
            # ここで、前置詞の除去も行われる
            splitted_by_prep = re.split(PATTERN, category)
            # 分割した後の左側を単語単位に分割する
            splitted_first_category = splitted_by_prep[0].split()
            for w in splitted_first_category:
                new_categories.append(w)

            for i in range(1, len(splitted_by_prep)):
                new_categories.append(splitted_by_prep[i])
        return new_categories

def main():
    with open('configs/log_config.json', 'r') as f:
        log_conf = json.load(f)
    config.dictConfig(log_conf)
    logger = getLogger(__name__)

    # 隠しカテゴリ・追跡カテゴリのファイルを読み込み、リストを作る
    ht_categories = []
    with open(HT_FILE_PATH) as ht_file:
        for line in ht_file:
            line = line.replace('http://dbpedia.org/resource/Category:', "") # 「Archdeacons_of_Raphoe」
            line = line.replace("_", " ") # 「Archdeacons of Raphoe」
            line = line.lower() # 「archdeacons of raphoe」
            ht_categories.append(line)

    counter = 0
    with gzip.open(CIRRUSSEARCH_FILE_PATH) as file:
        with open(OUTPUT_FILE_PATH, "w") as output_file:
            # cirrus dumpから1行読み込んで、必要なものだけをクラスに入れ、1つのインスタンスを作る
            for line in tqdm.tqdm(file):
                data: dict = json.loads(line)
                if "index" not in data:
                    counter += 1
                    if ("debugpy" in sys.modules and counter > 1000):
                        break
                    try:
                        preprocessed_data = PreProcessedCirrusDump(ht_categories, data["title"], data["heading"], data["category"], data.get("ores_articletopics"), data.get("ores_articletopic"))
                    except Exception as e:
                        logger.error("{0}, {1}".format(type(e), e))
                    else:
                        # そのインスタンスをシリアライズして、ファイルに書き込む
                        json_str = json.dumps(vars(preprocessed_data))
                        output_file.write(json_str + "\n")

if __name__ == '__main__':
    main()