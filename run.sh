rm -r cache_dir
python finetune.py electra-base

rm -r cache_dir
python finetune.py bert

rm -r cache_dir
python finetune.py roberta

rm -r cache_dir
python finetune.py distilbert

rm -r cache_dir
python finetune.py distilroberta

rm -r cache_dir
python finetune.py xlnet