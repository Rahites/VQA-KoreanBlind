# KoMETER-V (KLUE-RoBERTa version)
Base Architecture : https://github.com/zdou0830/METER  
KLUE-RoBERTa : https://huggingface.co/klue/roberta-base  

## Environment
pip install -r requirements.txt  
pip install -e .

## Data Format
https://github.com/dandelin/ViLT/blob/master/DATA.md  
Pretrain : COCO Format  
Finetuning : VQAv2 Format

## Make arrow data
- make_custom_data_arrow : Make Pretrain Arrow Data
- make_custom_data_arrow_vqa1 : Make FineTuning Arrow Data
- make_custom_data_arrow_vqa1_nofilter : Make FineTuning Nofilter Arrow Data

**Change Compared Original**
1. meter/datasets/_\_init__.py
2. meter/datamodules/_\_init__.py  
3. meter/datasets/coco_caption_korean_dataset.py
4. meter/datasets/vqav2_dataset_v1.py
5. meter/datamodules/coco_caption_korean_datamodule.py
6. meter/datamodules/vqav2_datamodule_v1.py
7. meter/config.py 
8. meter/modules/meter_module.py
9. meter/modules/bert_model.py
10. meter/modules/objectives.py
11. meter/datamodules/datamodule_base.py

## Pretrain
```
python run.py with data_root=<arrow_data_path> num_gpus=1 num_nodes=1 task_mlm_itm_clip_bert per_gpu_batchsize=6 clip16 text_klue image_size=288
```

## FineTuning
```
python run.py with data_root=<arrow_data_path> num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=12 load_path=<Pretrain_ckpt_path> clip16 text_klue image_size=288 clip_randaug
```
## Evaluation

```
python run.py with data_root=<arrow_data_path> num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=12 load_path=<FineTuned_ckpt_path> clip16 text_klue image_size=288 test_only=True
```