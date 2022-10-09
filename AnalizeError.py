import json
import tqdm
import argparse
import blink.main_dense as main_dense
from logging import getLogger, config
from PreProcessCirrusDump import RoughenCategory

TEST_DATA_PATH = "../Dataset/unseen_mentions/test.json"
CIRRUSSEARCH_FILE_PATH = "outputs/preprocessed_dumps.json"
OUTPUT_FILE_PATH = "outputs/analizeError_result.txt"
LOW_RECALL_THRESHOLD = 80
LOW_RECALL_TAG = "[LOW_RECALL]"
BLINKS_OUTPUT_DOSENT_APPEAR_IN_CIRRSU_DUMPS = "BLINK'S OUTPUT DOSEN'T APPEAR IN CIRRSU DUMPS!"
BLINKS_ALL_OUTPUTS_DONT_APPEAR_IN_CIRRSU_DUMPS = "BLINK'S ALL OUTPUTS DON'T APPEAR IN CIRRSU DUMPS!"
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

def main():
    with open('configs/log_config.json', 'r') as f:
        log_conf = json.load(f)
    config.dictConfig(log_conf)
    logger = getLogger(__name__)

    logger.info("--------------------------------------------------------------------------------------------------")
    logger.info("Start!")
    logger.info("事前処理済みcirrus dumpの読み込み")
    preprocessed_cirrus_dumps = []
    with open(CIRRUSSEARCH_FILE_PATH) as preprocessed_cirrus_file:
        for line in tqdm.tqdm(preprocessed_cirrus_file):
            preprocessed_cirrus_dump: dict = json.loads(line)
            preprocessed_cirrus_dumps.append(preprocessed_cirrus_dump)

    args = argparse.Namespace(**blink_config)
    logger.info("BLINKモデル読み込み開始")
    models = main_dense.load_models(args, logger=None)
    logger.info("BLINKモデル読み込み終了")

    logger.info("元のtest.jsonを1行ずつ読み込む（1万行）")
    with open(OUTPUT_FILE_PATH, "w") as output_file:
        with open(TEST_DATA_PATH) as test_data_file:
            id = 0
            sum_no_match_in_cirrsu_dumps = 0
            sum_match = 0.0
            sum_recall = 0.0
            for line in tqdm.tqdm(test_data_file):
                output_file.write(f"----------id:{id:06}---------------------------------------------------------------------------------------------\n")
                test_data: dict = json.loads(line)
                output_file.write(f"[test.json] docId:{test_data['docId']}\n")
                output_file.write(f"[test.json] wikiurl:{test_data['wikiurl']}\n")
                output_file.write(f"[test.json] wikiId:{test_data['wikiId']}\n")
                output_file.write(f"[test.json] y_title:{test_data['y_title']}\n")
                # 1行で書き込むとプログラムで読み込みやすいから、配列もそのままで
                output_file.write(f"[test.json] original_category:{test_data['y_category_original']}\n")

                # test.jsonから読み込んだカテゴリも粗くする必要がある
                lowered_categories = list(map(lambda c: c.lower(), test_data["y_category_original"]))
                test_data["roughen_category"] = RoughenCategory(lowered_categories)
                output_file.write(f"[test.json] roughen_category:{sorted(test_data['roughen_category'])}\n")

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

                match_score = 0.0
                sum_recall_per_instance = 0.0
                sum_no_match_per_instance = 0
                predicted_category = []
                # ダンプデータを使って、上の出力から、そのカテゴリを取得する
                for i in range(len(predictions[0])):
                    # 事前処理済みcirrus dumpリスト（len=100万ぐらい）において、BLINKが出力したtitleを持っている要素1つを抽出
                    founded_dump = [d for d in preprocessed_cirrus_dumps if ("title", predictions[0][i]) in d.items()]
                    if founded_dump: # dump内にtitleを持つwikipedia記事があった場合
                        predicted_category.append(founded_dump[0]["roughen_category"])
                    else:
                        predicted_category.append([])

                    output_file.write(f"[BLINK's {i+1}th output] ------------------------------------------------------------------------------------\n")
                    output_file.write(f"[BLINK's {i+1}th output] title              :{predictions[0][i]}\n")
                    output_file.write(f"[BLINK's {i+1}th output] score              :{scores[0][i]}\n")
                    if (predicted_category[i]):
                        output_file.write(f"[BLINK's {i+1}th output] roughen_category   :{sorted(predicted_category[i])}\n")
                    else:
                        logger.warning(f"id:{id}  BLINKが出力した{i+1}位のエンティティ「{predictions[0][i]}」が、cirrus dumpの中に含まれていません")
                        output_file.write(f"[BLINK's {i+1}th output] roughen_category   :{BLINKS_OUTPUT_DOSENT_APPEAR_IN_CIRRSU_DUMPS}\n")

                    # gold カテゴリと比較した結果を出す（具体的には、（粗くした）goldカテゴリの何割を、cirrus dumpから持ってきたカテゴリがカバーしてるかとか）
                    if (predictions[0][i] == test_data['y_title']):
                        # 順位の逆数分だけ、予測と実際の合致スコアを与える
                        match_score = 1 / (i+1)
                        sum_match += match_score
                        output_file.write(f"[BLINK's {i+1}th output] matches gold(test)?:[*YES*]\n")
                    else:
                        output_file.write(f"[BLINK's {i+1}th output] matches gold(test)?:[*NO *]\n")
                    if (predicted_category[i]):
                        try:
                            tmp_recall = len(set(predicted_category[i]) & set(test_data['roughen_category'])) / len(test_data['roughen_category']) * 100
                        except Exception as e:
                            logger.error(f"id:{id}  BLINKが出力した{i+1}位のエンティティ「{predictions[0][i]}」 / goldエンティティ「{test_data['y_title']}」 / 例外{type(e)}:{e}")
                            sum_no_match_per_instance += 1
                            sum_no_match_in_cirrsu_dumps += 1
                            output_file.write(f"[BLINK's {i+1}th output] recall(prediction & gold/gold(test) size):EXCEPTION({type(e)}) OCCURED!:{e}\n")
                        else:
                            output_file.write(f"[BLINK's {i+1}th output] recall(prediction & gold/gold(test) size):{tmp_recall:.3f}%")
                            output_file.write(f"{LOW_RECALL_TAG if tmp_recall < LOW_RECALL_THRESHOLD else ''}\n")
                            sum_recall_per_instance += tmp_recall
                            sum_recall += tmp_recall
                    else:
                        sum_no_match_per_instance += 1
                        sum_no_match_in_cirrsu_dumps += 1
                        output_file.write(f"[BLINK's {i+1}th output] recall(prediction & gold/gold(test) size):{BLINKS_OUTPUT_DOSENT_APPEAR_IN_CIRRSU_DUMPS}\n")
                output_file.write(f"[BLINK's all outputs] match gold(test):{match_score * 100:.3f}%\n")
                if (len(predictions[0]) == sum_no_match_per_instance):
                    output_file.write(f"[BLINK's all outputs] mean recall:{BLINKS_ALL_OUTPUTS_DONT_APPEAR_IN_CIRRSU_DUMPS}")
                else:
                    output_file.write(f"[BLINK's all outputs] mean recall:{sum_recall_per_instance / (len(predictions[0]) - sum_no_match_per_instance):.3f}%\n")
                output_file.write("----------------------------------------------------------------------------------------------------------------\n")
                id += 1

        logger.info("全体の結果を出力")
        output_file.write("----------total result------------------------------------------------------------------------------------------\n")
        output_file.write(f"[total result] mean BLINK's output matches gold(test):{sum_match / id * 100:.3f}%\n")
        output_file.write(f"[total result] mean recall(prediction & gold/gold(test) size):{sum_recall / (id*BLINK_CONFIG_TOP_K - sum_no_match_in_cirrsu_dumps):.3f}%\n")
    logger.info("Finished!")

if __name__ == '__main__':
    main()
