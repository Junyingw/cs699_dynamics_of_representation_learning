CUDA_VISIBLE_DEVICES=0 python create_directions.py \
--statefile_folder results/resnet20_relu_1024b/ckpt/ \
-r results/resnet20_relu_1024b --skip_bn_bias \
--direction_file pca_directions.npz --direction_style "pca" \
--model resnet20 \
--activation "relu" 

CUDA_VISIBLE_DEVICES=0 python create_directions.py \
--statefile_folder results/resnet20_tanh_1024b/ckpt/ \
-r results/resnet20_tanh_1024b --skip_bn_bias \
--direction_file pca_directions.npz --direction_style "pca" \
--model resnet20 \
--activation "tanh" 

CUDA_VISIBLE_DEVICES=0 python create_directions.py \
--statefile_folder results/resnet20_sigmoid_1024b/ckpt/ \
-r results/resnet20_sigmoid_1024b --skip_bn_bias \
--direction_file pca_directions.npz --direction_style "pca" \
--model resnet20 \
--activation "sigmoid" 

CUDA_VISIBLE_DEVICES=0 python create_directions.py \
--statefile_folder results/resnet20_siren_1024b/ckpt/ \
-r results/resnet20_siren_1024b --skip_bn_bias \
--direction_file pca_directions.npz --direction_style "pca" \
--model resnet20 \
--activation "siren" 