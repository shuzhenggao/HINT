gpu=0




lang=assert
#*******************************
#Parameters of SSL
mode=pl_ours2
threshold=0.4
percent=0.35
k=0.5
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base/
data_dir=/resource/dataset/atlas_assert/Datasets/Raw_Dataset/
num_train_epochs=10
train_batch_size=16
eval_batch_size=28
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/train_ssl_processed.jsonl
dev_file=$data_dir/valid_processed.jsonl
unlabel_file=$data_dir/unlabel_processed.jsonl 
load_model_path=model/$lang/pl_ours1/checkpoint-best-bleu/pytorch_model.bin
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent





lang=assert
#*******************************
#Parameters of SSL
mode=pl_ours3
threshold=0.4
percent=0.35
k=0.5
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base/
data_dir=/resource/dataset/atlas_assert/Datasets/Raw_Dataset/
num_train_epochs=10
train_batch_size=16
eval_batch_size=28
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/train_ssl_processed.jsonl
dev_file=$data_dir/valid_processed.jsonl
unlabel_file=$data_dir/unlabel_processed.jsonl 
load_model_path=model/$lang/pl_ours2/checkpoint-best-bleu/pytorch_model.bin
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent




lang=assert
#*******************************
#Parameters of SSL
mode=pl_ours4
threshold=0.4
percent=0.35
k=0.5
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base/
data_dir=/resource/dataset/atlas_assert/Datasets/Raw_Dataset/
num_train_epochs=10
train_batch_size=16
eval_batch_size=28
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/train_ssl_processed.jsonl
dev_file=$data_dir/valid_processed.jsonl
unlabel_file=$data_dir/unlabel_processed.jsonl 
load_model_path=model/$lang/pl_ours3/checkpoint-best-bleu/pytorch_model.bin
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent




lang=assert
#*******************************
#Parameters of SSL
mode=pl_ours5
threshold=0.4
percent=0.35
k=0.5
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base/
data_dir=/resource/dataset/atlas_assert/Datasets/Raw_Dataset/
num_train_epochs=10
train_batch_size=16
eval_batch_size=28
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/train_ssl_processed.jsonl
dev_file=$data_dir/valid_processed.jsonl
unlabel_file=$data_dir/unlabel_processed.jsonl 
load_model_path=model/$lang/pl_ours4/checkpoint-best-bleu/pytorch_model.bin
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent




lang=assert
#*******************************
#Parameters of SSL
mode=pl_ours6
threshold=0.4
percent=0.35
k=0.5
#*******************************
pretrained_model=/resource/pretrain/Salesforce/codet5-base/
data_dir=/resource/dataset/atlas_assert/Datasets/Raw_Dataset/
num_train_epochs=10
train_batch_size=16
eval_batch_size=28
beam_size=3
output_dir=model/$lang/$mode
train_file=$data_dir/train_ssl_processed.jsonl
dev_file=$data_dir/valid_processed.jsonl
unlabel_file=$data_dir/unlabel_processed.jsonl 
load_model_path=model/$lang/pl_ours5/checkpoint-best-bleu/pytorch_model.bin
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_train --do_eval --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent
python run.py --visible_gpu $gpu --lang $lang --max_source_length 256 --max_target_length 128  --train_file $train_file --dev_file $dev_file \
              --do_test --model_name $pretrained_model --output_dir $output_dir --load_model_path $load_model_path\
              --num_train_epochs $num_train_epochs --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --beam_size $beam_size \
              --mode $mode --unlabel_filename $unlabel_file --threshold $threshold --k $k --percent $percent
