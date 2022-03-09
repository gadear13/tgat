# Temporal Graph Neural Network #

Readme in construction

## Requirements ##
- pandas
- numpy
- torch
- torchmetrics
- Pytorch-geometric
- Pytorch-lightning

## Example ##

`python main.py --model_name TGATv3 --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --gradient_clip_val 0 --batch_size 64 --hid_channels 128 --heads 2 --dropout 0.1 --loss mse_with_regularizer --seq_len 4 --data m30 --split_ratio 0.8 --result_path results  --settings gat --log_path logs  --gpus 2 --pre_len `
