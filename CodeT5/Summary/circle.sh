gpu=0


lang=python
#*******************************
#Parameters of SSL
mode=codesum_pl_ours2
k=0.5
threshold=0.4
percent=0.25
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base
data_dir=/resource/dataset/CodeSumData/
num_train_epochs=10
train_batch_size=32
eval_batch_size=32
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/$lang/train_ssl.jsonl
dev_file=$data_dir/$lang/valid.jsonl
unlabel_file=$data_dir/$lang/unlabel.jsonl 
load_model_path=model/$lang/codesum_pl_ours1/checkpoint-best-bleu/pytorch_model.bin

#train
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent  
#generate
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/test.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model\
              --eval_batch_size $eval_batch_size --beam_size $beam_size --k $k --percent $percent  


lang=python
#*******************************
#Parameters of SSL
mode=codesum_pl_ours3
k=0.5
threshold=0.4
percent=0.25
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base
data_dir=/resource/dataset/CodeSumData/
num_train_epochs=10
train_batch_size=32
eval_batch_size=32
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/$lang/train_ssl.jsonl
dev_file=$data_dir/$lang/valid.jsonl
unlabel_file=$data_dir/$lang/unlabel.jsonl 
load_model_path=model/$lang/codesum_pl_ours2/checkpoint-best-bleu/pytorch_model.bin

#train
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent  
#generate
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/test.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model\
              --eval_batch_size $eval_batch_size --beam_size $beam_size --k $k --percent $percent  


lang=python
#*******************************
#Parameters of SSL
mode=codesum_pl_ours4
k=0.5
threshold=0.4
percent=0.25
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base
data_dir=/resource/dataset/CodeSumData/
num_train_epochs=10
train_batch_size=32
eval_batch_size=32
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/$lang/train_ssl.jsonl
dev_file=$data_dir/$lang/valid.jsonl
unlabel_file=$data_dir/$lang/unlabel.jsonl 
load_model_path=model/$lang/codesum_pl_ours3/checkpoint-best-bleu/pytorch_model.bin

#train
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent  
#generate
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/test.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model\
              --eval_batch_size $eval_batch_size --beam_size $beam_size --k $k --percent $percent  


lang=python
#*******************************
#Parameters of SSL
mode=codesum_pl_ours5
k=0.5
threshold=0.4
percent=0.25
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base
data_dir=/resource/dataset/CodeSumData/
num_train_epochs=10
train_batch_size=32
eval_batch_size=32
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/$lang/train_ssl.jsonl
dev_file=$data_dir/$lang/valid.jsonl
unlabel_file=$data_dir/$lang/unlabel.jsonl 
load_model_path=model/$lang/codesum_pl_ours4/checkpoint-best-bleu/pytorch_model.bin

#train
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent  
#generate
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/test.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model\
              --eval_batch_size $eval_batch_size --beam_size $beam_size --k $k --percent $percent  


lang=python
#*******************************
#Parameters of SSL
mode=codesum_pl_ours6
k=0.5
threshold=0.4
percent=0.25
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base
data_dir=/resource/dataset/CodeSumData/
num_train_epochs=10
train_batch_size=32
eval_batch_size=32
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/$lang/train_ssl.jsonl
dev_file=$data_dir/$lang/valid.jsonl
unlabel_file=$data_dir/$lang/unlabel.jsonl 
load_model_path=model/$lang/codesum_pl_ours5/checkpoint-best-bleu/pytorch_model.bin

#train
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent  
#generate
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
test_file=$data_dir/$lang/test.jsonl
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --mlm_probability 0.1 --test_file $test_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $test_model\
              --eval_batch_size $eval_batch_size --beam_size $beam_size --k $k --percent $percent  