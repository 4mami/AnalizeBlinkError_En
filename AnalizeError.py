from __future__ import annotations
import argparse
import json
import blink.main_dense as main_dense
import tqdm
from MyLogger import MyLogger
from PreProcessCirrusDump import RoughenCategory

class Program:
    TEST_DATA_PATH = "../Dataset/unseen_mentions/test.json"
    CIRRUSSEARCH_FILE_PATH = "outputs/preprocessed_dumps.json"
    OUTPUT_FILE_PATH = "outputs/ApplyBLINK.json"
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

    def run(self):
        logger = MyLogger.initialize(__name__, "configs/log_config.json")
        logger.info("--------------------------------------------------------------------------------------------------")
        logger.info("Start!")
        logger.info("事前処理済みcirrus dumpの読み込み")
        preprocessed_cirrus_dumps = self.load_preprocessed_cirrus_dumps(self.CIRRUSSEARCH_FILE_PATH)

        args = argparse.Namespace(**self.blink_config)
        logger.info("BLINKモデル読み込み開始")
        models = main_dense.load_models(args, logger=None)
        logger.info("BLINKモデル読み込み終了")

        logger.info("元のtest.jsonを1行ずつ読み込む（1万行）")
        with open(self.TEST_DATA_PATH) as test_data_file:
            id = 0
            for line in tqdm.tqdm(test_data_file):
                test_data: dict = json.loads(line)
                # test.jsonから読み込んだカテゴリも粗くする必要がある
                lowered_categories = list(map(lambda c: c.lower(), test_data["y_category_original"]))
                test_data["roughen_category"] = RoughenCategory(lowered_categories)

                # 各行のデータをBLINKに読ませて、出力（Wikipediaタイトル名）を得る
                data_to_link = [ {
                        "id": id,
                        "label": "unknown",
                        "label_id": -1,
                        "context_left": test_data["left_context_text"].lower(),
                        "mention": test_data["word"].lower(),
                        "context_right": test_data["right_context_text"].lower(),
                    } ]
                _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
                scores_float = list(map(lambda s: float(s), scores[0]))

                founded_dumps = []
                # ダンプデータを使って、上の出力から、そのカテゴリを取得する
                for i in range(len(predictions[0])):
                    # 事前処理済みcirrus dumpリスト（len=100万ぐらい）において、BLINKが出力したtitleを持っている要素1つを抽出
                    founded_dump = [d for d in preprocessed_cirrus_dumps if ("title", predictions[0][i]) in d.items()]
                    # dump内にtitleを持つwikipedia記事があった場合
                    if founded_dump:
                        # founded_dumpは抽出結果のリスト(len=1)なので、1番目にアクセスする
                        founded_dumps.append(founded_dump[0])
                    else:
                        logger.warning(f"id:{id}  BLINKが出力した{i+1}位のエンティティ「{predictions[0][i]}」が、dumpの中に含まれていません")
                        founded_dumps.append({})

                output_dict = self.make_output_dict(id, test_data, list(zip(predictions[0], scores_float)), founded_dumps)
                json_str = json.dumps(output_dict)
                with open(self.OUTPUT_FILE_PATH, "a") as output_file:
                    output_file.write(json_str)
                    output_file.write("\n")
                id += 1
        logger.info("Finished!")

    def load_preprocessed_cirrus_dumps(self, cirrussearch_file_path: str):
        ret = []
        with open(cirrussearch_file_path) as preprocessed_cirrus_file:
            for line in tqdm.tqdm(preprocessed_cirrus_file):
                preprocessed_cirrus_dump: dict = json.loads(line)
                ret.append(preprocessed_cirrus_dump)
        return ret

    def make_output_dict(self, id: int, test_data: dict, blinks_outputs: list[tuple], founded_dumps: list[dict]) -> dict:
        """_summary_

        :param int id: この実験上のID
        :param dict test_data: test.json内の単一の辞書オブジェクト（ラフにしたカテゴリを含む）
        :param list[tuple] blinks_outputs: BLINKの結果・スコアの配列
        :param list[dict] founded_dumps: BLINK出力とタイトルが適合したdump内の配列（タイトルとカテゴリとラフにしたカテゴリを含む）
        :return dict: 上を一緒にした辞書オブジェクト
        """
        test_data["experimentId"] = id
        for i in range(len(blinks_outputs)):
            test_data[f"BLINK_{i+1}th_output"] = {}
            test_data[f"BLINK_{i+1}th_output"]["BLINK_title"] = blinks_outputs[i][0]
            test_data[f"BLINK_{i+1}th_output"]["BLINK_score"] = blinks_outputs[i][1]
            test_data[f"BLINK_{i+1}th_output"].update(founded_dumps[i])
        return test_data

if __name__ == '__main__':
    Program().run()
