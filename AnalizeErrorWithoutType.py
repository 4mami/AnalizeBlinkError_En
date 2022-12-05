from __future__ import annotations
import argparse
import json
import blink.main_dense as main_dense
import tqdm
from MyLogger import MyLogger

class Program:
    DATASET_DIR = "../Dataset/"
    TEST_DATA_UNSEEN_MENTIONS = "unseen_mentions/test.json"
    TEST_DATA_KORE50 = "kore50-lrec2020/kore50-nif-dbpedia.ttl"
    # TODO: zeshelも読み込む
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

        logger.info(f"{self.TEST_DATA_UNSEEN_MENTIONS}を1行ずつ読み込む（1万行）")
        with open(self.TEST_DATA_UNSEEN_MENTIONS) as test_data_file:
            for line in tqdm.tqdm(test_data_file):
                test_data, data_to_link = self.unseen_mentions_line_to_blink_input(self.id, line)
                _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
                scores_float = list(map(lambda s: float(s), scores[0]))

                output_dict = self.make_output_dict(self.id, test_data, list(zip(predictions[0], scores_float)))
                self.write_output(output_dict)
                self.id += 1

        logger.info(f"{self.TEST_DATA_KORE50}を1行ずつ読み込む（1560行）")
        with open(self.TEST_DATA_KORE50, encoding="ISO-8859-1") as test_data_file:
            input_blink: bool = True
            sentence: str = ""
            mention: str = ""
            url: str = ""
            for line in tqdm.tqdm(test_data_file):
                # lineが「	nif:isString 」から始まっているなら
                    input_blink = True
                    # lineの左端をstrip()して、「"」で分割する
                    # 分割結果が3じゃなかったら、ログにエラーを書いてフラグをFalseに
                    # 分割結果リストの2番目をsentenceに保存
                # lineが「	nif:anchorOf 」から始まっているなら
                    # lineの左端をstrip()して、「"」で分割する
                    # 分割結果が3じゃなかったら、ログにエラーを書いてフラグをFalseに
                    # 分割結果リストの2番目をmentionに保存
                # lineが「	itsrdf:taIdentRef  」から始まっているなら
                    # lineの左端をstrip()して、「< or >」で分割する
                    # 分割結果が3じゃなかったら、ログにエラーを書いてフラグをFalseに
                    # 分割結果リストの2番目をurlに保存

                    if (input_blink):
                        # urlを「/」で分割した最後をtmp_goldに保持
                        # sentenceをmentionで分割した結果をtmp_contextに保存
                        test_data = dict()
                        test_data["word"] = mention
                        test_data["left_context_text"] = tmp_context[0] # 分割した結果が空文字列になってもリストの要素になるんだっけ？
                        test_data["right_context_text"] = tmp_context[1]
                        test_data["y_title"] = tmp_gold
                        data_to_link = [ {"id": id, "label": "unknown", "label_id": -1, "context_left": test_data["left_context_text"].lower(), "mention": test_data["word"].lower(), "context_right": test_data["right_context_text"].lower()} ]
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
