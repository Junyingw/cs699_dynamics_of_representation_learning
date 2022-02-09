CUDA_VISIBLE_DEVICES=0 python compute_trajectory.py \
-r results/resnet20_relu_1024b/trajectories \
--direction_file results/resnet20_relu_1024b/pca_directions.npz \
--projection_file pca_dir_proj.npz --model resnet20  --remove_skip_connections \
-s results/resnet20_relu_1024b/ckpt --skip_bn_bias \
--activation "relu" 

CUDA_VISIBLE_DEVICES=0 python compute_trajectory.py \
-r results/resnet20_tanh_1024b/trajectories \
--direction_file results/resnet20_tanh_1024b/pca_directions.npz \
--projection_file pca_dir_proj.npz --model resnet20  --remove_skip_connections \
-s results/resnet20_tanh_1024b/ckpt --skip_bn_bias \
--activation "tanh"

CUDA_VISIBLE_DEVICES=0 python compute_trajectory.py \
-r results/resnet20_sigmoid_1024b/trajectories \
--direction_file results/resnet20_sigmoid_1024b/pca_directions.npz \
--projection_file pca_dir_proj.npz --model resnet20  --remove_skip_connections \
-s results/resnet20_sigmoid_1024b/ckpt --skip_bn_bias \
--activation "sigmoid"

CUDA_VISIBLE_DEVICES=0 python compute_trajectory.py \
-r results/resnet20_siren_1024b/trajectories \
--direction_file results/resnet20_siren_1024b/pca_directions.npz \
--projection_file pca_dir_proj.npz --model resnet20  --remove_skip_connections \
-s results/resnet20_siren_1024b/ckpt --skip_bn_bias \
--activation "siren"

