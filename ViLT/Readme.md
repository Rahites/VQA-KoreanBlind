# ViLT
https://github.com/dandelin/ViLT

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

**Change Compared Original**
1. vilt/datasets/_\_init__.py
2. vilt/datamodules/_\_init__.py  
3. vilt/datasets/coco_caption_korean_dataset.py
4. vilt/datasets/vqav2_dataset_v1.py
5. vilt/datamodules/coco_caption_korean_datamodule.py
6. vilt/datamodules/vqav2_datamodule_v1.py
7. vilt/config.py 

## Pretrain
```
python run.py with data_root=<arrow_data_path> num_gpus=1 num_nodes=1 task_mlm_itm whole_word_masking=True step100k per_gpu_batchsize=24 exp_name=pretrain 
```

## FineTuning
```
python run.py with data_root=<arrow_data_path> num_gpus=1 num_nodes=1 task_finetune_vqa_randaug per_gpu_batchsize=32 load_path="<Pretrain_ckpt_path>"
```
## Evaluation

```
python run.py with data_root=<arrow_data_path> num_gpus=1 num_nodes=1 per_gpu_batchsize=32 task_finetune_vqa_randaug test_only=True precision=32 load_path="<FineTuned_ckpt_path>"
```