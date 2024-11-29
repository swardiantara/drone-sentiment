rm -r cache_dir
python finetune.py --model_type bert --model_name_or_path bert-base-cased

rm -r cache_dir
python finetune.py --model_type roberta --model_name_or_path roberta-base

rm -r cache_dir
python finetune.py --model_type distilbert --model_name_or_path distilbert-base-cased

rm -r cache_dir
python finetune.py --model_type roberta --model_name_or_path distilroberta-base

rm -r cache_dir
python finetune.py --model_type xlnet --model_name_or_path xlnet-base-cased

rm -r cache_dir
python finetune.py --model_type albert --model_name_or_path albert/albert-base-v2

rm -r cache_dir
python finetune.py --model_type electra --model_name_or_path google/electra-base-discriminator