CUDA_VISIBLE_DEVICES=0 python train.py \
--mode train \
--model resnet56 \
--batch_size 128 \
--skip_bn_bias \
--remove_skip_connections \
--activation "siren" \
--result_folder "results/resnet56_siren/" 

# CUDA_VISIBLE_DEVICES=0 python train.py \
# --mode train \
# --model resnet56 \
# --batch_size 128 \
# --skip_bn_bias \
# --remove_skip_connections \
# --activation "relu" \
# --result_folder "results/resnet56_relu/" 

# CUDA_VISIBLE_DEVICES=0 python plot.py --result_folder figures/resnet56/ \
# --trajectory_file results/resnet56_skip_bn_bias/trajectories/pca_dir_proj.npz \
# --surface_file results/resnet56_skip_bn_bias/loss_surface/pca_dir_loss_surface.npz \
# --plot_prefix resnet56_pca_dir