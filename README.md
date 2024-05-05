# NLLB with HQQ
This is an example repository to convert Transformers-based NLLB models and run inference with HQQ Optimization. Learn more about the HQQ project via their [official repository](https://github.com/mobiusml/hqq).

## Requirements
For this to work, you need to install the latest transformers dependency from github:
```txt
git+https://github.com/huggingface/transformers
```
Though, if you don't wish to create a python virtual environment, you could also opt to build the docker image via the provided `Dockerfile`. Simply run:
```sh
docker build -t dleongsh/nllb-with-hqq:0.0.1 .
```
Then start up the docker container via the `docker-compose.yaml` file and enter the interactive bash terminal within.
```sh
docker-compose run --rm nllb bash
```

## Conversion of Model Weights
This repository assumes that you already have the NLLB model weights and files (in HuggingFace Transformers format).
This repository also assumes you have the tokenizer files downloaded as well. Put them in the tokenizer folder. Your final input directory should look like this:
```
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
```

## Test Inference
In `src/main.py`, switch up these variables for your own.
```
MODEL_PATH = "/pretrained_models/nllb-200-distilled-600m-lora-ct2-f16-zh"
TEXT = "我很可爱，你知道吗？"
SOURCE_LANGUAGE = "zho_Hans"
TARGET_LANGUAGE = "eng_Latn"
```
Then run it. That's all~
