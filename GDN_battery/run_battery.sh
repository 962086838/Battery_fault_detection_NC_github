gpu_n=$1
DATASET=$2
fold_num=$3
val_loader=$4
EPOCH=$5

seed=5
BATCH_SIZE=128
SLIDE_WIN=32
dim=64
out_layer_num=1
SLIDE_STRIDE=16
topk=5
out_layer_inter_dim=128
val_ratio=0
decay=0



path_pattern="${DATASET}"
COMMENT="${DATASET}"

report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python3 main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python3 main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        --fold_num $fold_num \
        --val_loader $val_loader
fi