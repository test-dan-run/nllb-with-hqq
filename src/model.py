""" MT Model Class """
import os
import logging
from time import perf_counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, HqqConfig

# pylint: disable=too-few-public-methods
class MTModel:
    """MT Model instance"""

    def __init__(self, model_dir: str, source_language: str, target_language: str):
        """
        model_dir (str):
            Directory where your model and tokenizer files are stored
            File directory should look like this:
                |- model_dir
                |   |- model
                |   |   |- config.json
                |   |   |- model.bin
                |   |   |- shared_vocabulary.json
                |   |- tokenizer
                |   |   |- sentencepiece.bpe.model
                |   |   |- special_tokens_map.json
                |   |   |- tokenizer_config.json
                |   |   |- tokenizer.json

        source_language (str): Language to translate from, uses FLORES-200 language code
        target_language (str): Language to translate into, uses FLORES-200 language code
        """
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        logging.info("Running on device: %s", self.device)

        logging.info("Loading model...")
        model_load_start = perf_counter()

        self.source_language = source_language
        self.target_language = target_language

        tokenizer_path = os.path.join(model_dir, "tokenizer")
        assert os.path.exists(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer"),
            src_lang=self.source_language,
            tgt_lang=self.target_language,
        )

        hqq_config = HqqConfig(
            nbits=8,
            group_size=32,
            quant_zero=False,
            quant_scale=False,
            axis=0,
            offload_meta=False
        )  # axis=0 is used by default

        model_path = os.path.join(model_dir, "model")
        assert os.path.exists(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            quantization_config=hqq_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "12GiB"},
            device_map="auto")

        model_load_end = perf_counter()
        logging.info(
            "Model loaded. Elapsed time: %s", model_load_end - model_load_start
        )

    def _translate(self, input_text: str) -> str:
        """Takes in a text and translate Language X -> English
        referenced: https://opennmt.net/CTranslate2/guides/transformers.html#nllb
        """

        with torch.no_grad():
            input_tokens = self.tokenizer(
                [input_text], return_tensors="pt", padding=True
            ).to(self.device)

            output_tokens = self.model.generate(
                **input_tokens,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"],
                max_new_tokens=128,
            )
            output_text = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True
            )[0]

        return output_text

    # pylint: disable=unused-argument,no-member
    def translate(self, text: str) -> str:
        """ Wraps the internal method. Replace this method with a wrapper for your service call"""

        infer_start = perf_counter()
        output_text = self._translate(text)
        infer_end = perf_counter()
        logging.info("Inference elapsed time: %s", infer_end - infer_start)

        return output_text
