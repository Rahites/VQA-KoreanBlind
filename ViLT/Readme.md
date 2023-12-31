# ViLT
https://github.com/dandelin/ViLT

## Make arrow data
- make_custom_data_arrow : Make Pretrain Arrow Data
- make_custom_data_arrow_vqa1 : Make FineTuning Arrow Data


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