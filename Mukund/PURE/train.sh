#!/bin/bash
#change working directory

#download dataset
pip install gdown
gdown https://drive.google.com/uc?id=1Ls8bNSWVbZ2HZiXsZdnh8HL01NY6yk9H
gdown https://drive.google.com/uc?id=1rq0MDeP93LXpVeYh3GN48JhqBeaWR7Vc
unzip ace05.zip
unzip data.zip -d DyGIE/
rm ace05.zip
rm data.zip

#install dependencies
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu112/torch_stable.html
pip install -r requirements.txt
pip install higher

#run training code
python run_relation.py \
  --task ace05 \
  --do_meta_train --train_file DyGIE/data/ace05/json/train.json \
  --do_eval --eval_test --eval_with_gold \
  --model bert-base-uncased \
  --do_lower_case \
  --train_batch_size 1 \
  --eval_batch_size 16 \
  --eval_per_epoch 2 \
  --inner_learning_rate 1e-4\
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --context_window 0 \
  --max_seq_length 128 \
  --entity_output_dir DyGIE/data/ace05/json \
  --entity_predictions_dev dev.json \
  --entity_predictions_test test.json \
  --output_dir ace05/trainedModel/