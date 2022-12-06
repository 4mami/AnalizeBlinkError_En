from __future__ import annotations
import argparse
import json
import re
import xml.etree.ElementTree as ET
import blink.main_dense as main_dense
import tqdm
from MyLogger import MyLogger

class Program:
    DATASET_DIR = "../Dataset/"
    TEST_DATA_UNSEEN_MENTIONS = "unseen_mentions/test.json"
    TEST_DATA_KORE50 = "kore50-lrec2020/kore50-nif-dbpedia.ttl"
    TEST_DATA_YAHOO = "yahoo_webscope_L24/ydata-search-query-log-to-entities-v1_0.xml"
    OUTPUT_FILE_PATH = "outputs/ApplyBLINKWitoutType.json"
    BLINK_CONFIG_TOP_K = 5

    models_path = "../BLINK/models/"
    blink_config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": BLINK_CONFIG_TOP_K,
        "biencoder_model": models_path+"biencoder_wiki_large.bin",
        "biencoder_config": models_path+"biencoder_wiki_large.json",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
        "crossencoder_config": models_path+"crossencoder_wiki_large.json",
        "fast": False,
        "output_path": "outputs/"
    }
    id = 0

    def run(self):
        logger = MyLogger.initialize(__name__, "configs/log_config.json")
        logger.info("--------------------------------------------------------------------------------------------------")
        logger.info("Start!")

        args = argparse.Namespace(**self.blink_config)
        logger.info("BLINKモデル読み込み開始")
        models = main_dense.load_models(args, logger=None)
        logger.info("BLINKモデル読み込み終了")

        logger.info(f"{self.TEST_DATA_UNSEEN_MENTIONS}を1行ずつ読み込む（1万行） 現在id:{self.id}")
        with open(self.DATASET_DIR + self.TEST_DATA_UNSEEN_MENTIONS) as test_data_file:
            for line in tqdm.tqdm(test_data_file):
                test_data, data_to_link = self.unseen_mentions_line_to_blink_input(self.id, line)
                _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
                scores_float = list(map(lambda s: float(s), scores[0]))

                output_dict = self.make_output_dict(self.id, test_data, list(zip(predictions[0], scores_float)))
                self.write_output(output_dict)
                self.id += 1

        logger.info(f"{self.TEST_DATA_KORE50}を1行ずつ読み込む（1560行） 現在id:{self.id}")
        with open(self.DATASET_DIR + self.TEST_DATA_KORE50, encoding="ISO-8859-1") as test_data_file:
            sentence_list = list()
            mention_list = list()
            url_list = list()
            for line in tqdm.tqdm(test_data_file):
                if (re.match(r"\A\tnif:isString ", line)): # 	nif:isString "David and Victoria named their children Brooklyn, Romeo, Cruz, and Harper Seven."^^xsd:string .
                    sentence_list = line.split('"')
                elif (re.match(r"\A\tnif:anchorOf ", line)): # 	nif:anchorOf "David"^^xsd:string ;
                    mention_list = line.split('"')
                elif (re.match(r"\A\titsrdf:taIdentRef  ", line)): # 	itsrdf:taIdentRef  <http://dbpedia.org/resource/David_Beckham> .
                    url_list = re.split(r"<|>", line)
                    if (len(sentence_list) != 3 or len(mention_list) != 3 or len(url_list) != 3 or len(sentence_list[1].split(mention_list[1])) != 2):
                        logger.error(f"{self.TEST_DATA_KORE50}に不正な行: {line}")
                        continue
                    test_data, data_to_link = self.kore50_lines_to_blink_input(self.id, mention_list, sentence_list, url_list)
                        _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
                        scores_float = list(map(lambda s: float(s), scores[0]))

                        output_dict = self.make_output_dict(self.id, test_data, list(zip(predictions[0], scores_float)))
                        self.write_output(output_dict)
                        self.id += 1

        logger.info(f"{self.TEST_DATA_YAHOO}読み込み開始 現在id:{self.id}")
        tree = ET.parse(self.DATASET_DIR + self.TEST_DATA_YAHOO)
        root = tree.getroot()
        for session in tqdm.tqdm(root):
            for query in session:
                tmp_attr = query.attrib
                # 属性を見て、不適切なクエリならcontinue
                if (tmp_attr["ambiguous"] == "true" or tmp_attr["cannot-judge"] == "true" or tmp_attr["navigational"] == "true" or tmp_attr["no-wp"] == "true" or tmp_attr["non-english"] == "true" or tmp_attr["quote-question"] == "true"):
                    continue
                sentence_yahoo = ""
                for txt_and_anno in query:
                    if (txt_and_anno.tag == "text"):
                        sentence_yahoo = txt_and_anno.text
                    elif (txt_and_anno.tag == "annotation"):
                        tmp_mention = txt_and_anno[0].text
                        tmp_context = sentence_yahoo.split(tmp_mention)
                        if (len(tmp_context) != 2):
                            logger.error(f"{self.TEST_DATA_YAHOO}に不正な行: {session.get('id')}, {tmp_mention}")
                            continue
                        tmp_wikiurl = txt_and_anno[1].text
                        test_data, data_to_link = self.yahoo_annotation_node_to_blink_input(self.id, tmp_mention, tmp_context, tmp_wikiurl)
                        _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
                        scores_float = list(map(lambda s: float(s), scores[0]))

                        output_dict = self.make_output_dict(self.id, test_data, list(zip(predictions[0], scores_float)))
                        self.write_output(output_dict)
                        self.id += 1

        logger.info("Finished!")

    def unseen_mentions_line_to_blink_input(self, id: int, line: str):
        test_data: dict = json.loads(line)
        # 各行のデータをBLINKに読ませて、出力（Wikipediaタイトル名）を得る
        data_to_link = [ {
                "id": id,
                "label": "unknown",
                "label_id": -1,
                "context_left": test_data["left_context_text"].lower(),
                "mention": test_data["word"].lower(),
                "context_right": test_data["right_context_text"].lower(),
            } ]

        return test_data, data_to_link

    def kore50_lines_to_blink_input(self, id: int, mention_list: list, sentence_list: list, url_list: list):
        mention = mention_list[1]
        sentence = sentence_list[1]
        url = url_list[1]

        tmp_context = sentence.split(mention)
        test_data = dict()
        test_data["word"] = mention
        test_data["left_context_text"] = tmp_context[0].strip()
        test_data["right_context_text"] = tmp_context[1].strip()
        test_data["y_title"] = url.split("/")[-1].replace("_", " ")
        test_data["dbpediaurl"] = url # ここを改造して、wikiurlを作る
        data_to_link = [ {"id": id, "label": "unknown", "label_id": -1, "context_left": test_data["left_context_text"].lower(), "mention": test_data["word"].lower(), "context_right": test_data["right_context_text"].lower()} ]

        return test_data, data_to_link

    def yahoo_annotation_node_to_blink_input(self, id: int, mention: str, context: list, url :str):
        test_data = dict()
        test_data["word"] = mention
        test_data["left_context_text"] = context[0].strip()
        test_data["right_context_text"] = context[1].strip()
        test_data["y_title"] = url.split("/")[-1].replace("_", " ")
        test_data["wikiurl"] = url
        data_to_link = [ {"id": id, "label": "unknown", "label_id": -1, "context_left": test_data["left_context_text"].lower(), "mention": test_data["word"].lower(), "context_right": test_data["right_context_text"].lower()} ]

        return test_data, data_to_link

    def make_output_dict(self, id: int, test_data: dict, blinks_outputs: list[tuple]) -> dict:
        """_summary_

        :param int id: この実験上のID
        :param dict test_data: test.json内の単一の辞書オブジェクト
        :param list[tuple] blinks_outputs: BLINKの結果・スコアの配列
        :return dict: 上を一緒にした辞書オブジェクト
        """
        test_data["experimentId"] = id
        for i in range(len(blinks_outputs)):
            test_data[f"BLINK_{i+1}th_output"] = {}
            test_data[f"BLINK_{i+1}th_output"]["BLINK_title"] = blinks_outputs[i][0]
            test_data[f"BLINK_{i+1}th_output"]["BLINK_score"] = blinks_outputs[i][1]
        return test_data

    def write_output(self, output: dict):
        json_str = json.dumps(output)
        with open(self.OUTPUT_FILE_PATH, "a") as output_file:
            output_file.write(json_str)
            output_file.write("\n")

if __name__ == '__main__':
    Program().run()
