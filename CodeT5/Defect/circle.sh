export CUDA_VISIBLE_DEVICES=1



python run.py \
    --mode=defect_pl_ours_2\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_2 \
    --teacher_path=./saved_models/defect_pl_ours \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_train \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

python run.py \
    --mode=defect_pl_ours_2\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_2 \
    --teacher_path=./saved_models/defect_pl_ours \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_test \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log




python run.py \
    --mode=defect_pl_ours_3\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_3 \
    --teacher_path=./saved_models/defect_pl_ours_2 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_train \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

python run.py \
    --mode=defect_pl_ours_3\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_3 \
    --teacher_path=./saved_models/defect_pl_ours_2 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_test \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log







python run.py \
    --mode=defect_pl_ours_4\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_4 \
    --teacher_path=./saved_models/defect_pl_ours_3 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_train \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

python run.py \
    --mode=defect_pl_ours_4\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_4 \
    --teacher_path=./saved_models/defect_pl_ours_3 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_test \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log







python run.py \
    --mode=defect_pl_ours_5\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_5 \
    --teacher_path=./saved_models/defect_pl_ours_4 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_train \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

python run.py \
    --mode=defect_pl_ours_5\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_5 \
    --teacher_path=./saved_models/defect_pl_ours_4 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_test \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log



python run.py \
    --mode=defect_pl_ours_6\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_6 \
    --teacher_path=./saved_models/defect_pl_ours_5 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_train \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

python run.py \
    --mode=defect_pl_ours_6\
    --threshold=0.25\
    --edit=0.4\
    --k=0.5\
    --output_dir=./saved_models/defect_pl_ours_6 \
    --teacher_path=./saved_models/defect_pl_ours_5 \
    --model_type=roberta \
    --tokenizer_name=/resource/pretrain/salesforce/codet5-base \
    --model_name_or_path=/resource/pretrain/salesforce/codet5-base \
    --do_test \
    --train_data_file=/resource/dataset/defect/ssl/train_ssl.jsonl \
    --eval_data_file=/resource/dataset/defect/test.jsonl \
    --test_data_file=/resource/dataset/defect/test.jsonl \
    --unlabel_filename=/resource/dataset/defect/ssl/unlabel.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

