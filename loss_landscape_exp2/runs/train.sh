
# exp1 resnet20 relu 
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode train \
--model resnet20 \
--batch_size 1024 \
--skip_bn_bias \
--remove_skip_connections \
--activation "relu" \
--result_folder "results/resnet20_relu_1024/" 

# exp2 resnet20 tanh 
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode train \
--model resnet20 \
--batch_size 1024 \
--skip_bn_bias \
--remove_skip_connections \
--activation "tanh" \
--result_folder "results/resnet20_tanh_1024/" 

# exp3 resnet20 sigmoid 
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode train \
--model resnet20 \
--batch_size 1024 \
--skip_bn_bias \
--remove_skip_connections \
--activation "sigmoid" \
--result_folder "results/resnet20_sigmoid_1024/" 


# exp4 resnet20 siren
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode train \
--model resnet20 \
--batch_size 1024 \
--skip_bn_bias \
--remove_skip_connections \
--activation "siren" \
--result_folder "results/resnet20_siren_1024/" 