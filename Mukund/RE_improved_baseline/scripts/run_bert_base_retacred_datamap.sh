python train_retacred_datamap.py --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed 78 --train_batch_size 16 --test_batch_size 16 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base-retacred-datamap  --data_dir dataset/retacred --output_dir datamap_bert_retacred;