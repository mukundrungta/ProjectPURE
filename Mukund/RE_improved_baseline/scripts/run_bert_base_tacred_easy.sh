python train_tacred.py --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed 78 --train_batch_size 8 --test_batch_size 8 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base-tacred-easy  --data_dir dataset/tacred_filtered_easy;