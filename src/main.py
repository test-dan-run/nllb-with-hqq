import logging
from model import MTModel

MODEL_PATH = "/pretrained_models/nllb-200-distilled-600m-lora-ct2-f16"
TEXT = "我非常可爱哟"
SOURCE_LANGUAGE = "zho_Hans"
TARGET_LANGUAGE = "eng_Latn"

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s | %(message)s",
        level=logging.INFO
    )

    mt_model = MTModel(MODEL_PATH, SOURCE_LANGUAGE, TARGET_LANGUAGE)
    for i in range(20):
        translated_text = mt_model.translate(TEXT)
        logging.info("input : %s", TEXT)
        logging.info("output: %s", translated_text)
