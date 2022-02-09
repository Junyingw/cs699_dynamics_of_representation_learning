CUDA_VISIBLE_DEVICES=0 python plot.py \
--result_folder figures/resnet20_relu_1024b/ \
--trajectory_file results/resnet20_relu_1024b/trajectories/pca_dir_proj.npz \
--surface_file results/resnet20_relu_1024b/loss_surface/pca_dir_loss_surface.npz \
--plot_prefix resnet20_relu_1024b

CUDA_VISIBLE_DEVICES=0 python plot.py \
--result_folder figures/resnet20_tanh_1024b/ \
--trajectory_file results/resnet20_tanh_1024b/trajectories/pca_dir_proj.npz \
--surface_file results/resnet20_tanh_1024b/loss_surface/pca_dir_loss_surface.npz \
--plot_prefix resnet20_tanh_1024b

CUDA_VISIBLE_DEVICES=0 python plot.py \
--result_folder figures/resnet20_sigmoid_1024b/ \
--trajectory_file results/resnet20_sigmoid_1024b/trajectories/pca_dir_proj.npz \
--surface_file results/resnet20_sigmoid_1024b/loss_surface/pca_dir_loss_surface.npz \
--plot_prefix resnet20_sigmoid_1024b

# CUDA_VISIBLE_DEVICES=0 python plot.py \
# --result_folder figures/resnet20_siren_1024b/ \
# --trajectory_file results/resnet20_siren_1024b/trajectories/pca_dir_proj.npz \
# --surface_file results/resnet20_siren_1024b/loss_surface/pca_dir_loss_surface.npz \
# --plot_prefix resnet20_siren_1024b

