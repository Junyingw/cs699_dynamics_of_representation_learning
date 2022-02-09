# CUDA_VISIBLE_DEVICES=0 python compute_loss_surface.py \
# --result_folder results/resnet20_relu_1024b/loss_surface/  \
# -s results/resnet20_relu_1024b/ckpt/200_model.pt \
# --batch_size 1000 --skip_bn_bias \
# --model resnet20 --remove_skip_connections \
# --direction_file results/resnet20_relu_1024b/pca_directions.npz \
# --surface_file pca_dir_loss_surface.npz \
# --xcoords 51:-10:40 --ycoords 51:-10:40 \
# --activation "relu" 

# CUDA_VISIBLE_DEVICES=0 python compute_loss_surface.py \
# --result_folder results/resnet20_tanh_1024b/loss_surface/  \
# -s results/resnet20_tanh_1024b/ckpt/200_model.pt \
# --batch_size 1000 --skip_bn_bias \
# --model resnet20 --remove_skip_connections \
# --direction_file results/resnet20_tanh_1024b/pca_directions.npz \
# --surface_file pca_dir_loss_surface.npz \
# --xcoords 51:-10:40 --ycoords 51:-10:40 \
# --activation "tanh" 

# CUDA_VISIBLE_DEVICES=0 python compute_loss_surface.py \
# --result_folder results/resnet20_sigmoid_1024b/loss_surface/  \
# -s results/resnet20_sigmoid_1024b/ckpt/200_model.pt \
# --batch_size 1000 --skip_bn_bias \
# --model resnet20 --remove_skip_connections \
# --direction_file results/resnet20_sigmoid_1024b/pca_directions.npz \
# --surface_file pca_dir_loss_surface.npz \
# --xcoords 51:-10:40 --ycoords 51:-10:40 \
# --activation "sigmoid" 

CUDA_VISIBLE_DEVICES=0 python compute_loss_surface.py \
--result_folder results/resnet20_siren_1024b/loss_surface/  \
-s results/resnet20_siren_1024b/ckpt/200_model.pt \
--batch_size 1000 --skip_bn_bias \
--model resnet20 --remove_skip_connections \
--direction_file results/resnet20_siren_1024b/pca_directions.npz \
--surface_file pca_dir_loss_surface.npz \
--xcoords 51:-10:40 --ycoords 51:-10:40 \
--activation "siren" 



