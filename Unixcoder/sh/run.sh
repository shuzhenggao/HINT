lang=$1
threshold=$2
echo $lang
echo $threshold
mode=bsl
lr=5e-5
train_batch_size=32
eval_batch_size=128
beam_size=10
source_length=256
target_length=64
data_dir=CodeSumData
num_train_epochs=10
output_dir=model/$lang/$mode
train_file=$data_dir/$lang/train_ssl.jsonl
dev_file=$data_dir/$lang/valid.jsonl
unlabel_file=$data_dir/$lang/unlabel.jsonl
pretrained_model=microsoft/unixcoder-base
out_dir=checkpoint-bsl
python run_aug.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --lang $lang \
                  --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir\
                  --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size\
                  --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --learning_rate $lr --num_train_epochs $num_train_epochs \
                  --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --out_dir $out_dir

#Generate
test_model=$output_dir/$out_dir/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/test.jsonl
python run_ssl.py --do_eval --do_test --model_type roberta --model_name_or_path $pretrained_model --lang $lang \
                  --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir \
                  --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $eval_batch_size --out_dir $out_dir

mode=pl_ours
label_file=$output_dir/pseudo.jsonl
pseudo_file=pseudo-all.jsonl
load_model_path=$output_dir/$out_dir/pytorch_model.bin
out_dir=build
python run_aug.py --do_train  --model_type roberta --model_name_or_path $pretrained_model --lang $lang \
                    --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --load_model_path $load_model_path \
                    --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size\
                    --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --learning_rate $lr --num_train_epochs $num_train_epochs --out_dir $out_dir\
                    --mode $mode --unlabel_filename $unlabel_file --label_filename $label_file --pseudo_filename $pseudo_file     --submode 'build'

#retrain model 
out_dir=checkpoint-pl_ours
pseudo_file=pseudo-all.jsonl
selected_pseudo_file=pseudo-topk-$threshold.jsonl
python run_aug.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --lang $lang \
                  --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --load_model_path $load_model_path \
                  --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size\
                  --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --learning_rate $lr --num_train_epochs $num_train_epochs \
                  --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --pseudo_filename $pseudo_file --selected_pseudo_filename $selected_pseudo_file --out_dir $out_dir


#Generate
test_model=$output_dir/$out_dir/pytorch_model.bin #checkpoint for test
test_file=$data_dir/$lang/test.jsonl
python run_aug.py --do_eval --do_test --model_type roberta --model_name_or_path $pretrained_model  --lang $lang \
                  --load_model_path $test_model --dev_filename $test_file --test_filename $test_file --output_dir $output_dir \
                  --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $eval_batch_size --out_dir $out_dir
