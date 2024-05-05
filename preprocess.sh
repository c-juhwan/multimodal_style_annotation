clear

# python main.py --task=captioning --job=preprocessing --task_dataset=uit_viic_en_ori
# python main.py --task=visual_qa --job=preprocessing --task_dataset=uit_viic_en_ori

# python main.py --task=visual_entailment --job=preprocessing --task_dataset=snli_ve_sports_ori

python main.py --task=captioning --job=preprocessing --task_dataset=flickr8k
python main.py --task=captioning --job=preprocessing --task_dataset=flickr30k
python main.py --task=captioning --job=preprocessing --task_dataset=coco_karpathy
python main.py --task=visual_qa --job=preprocessing --task_dataset=vqa_v2
python main.py --task=visual_entailment --job=preprocessing --task_dataset=snli_ve

# download tar.gz file from https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl and move it to ./task/visual_entailment/
