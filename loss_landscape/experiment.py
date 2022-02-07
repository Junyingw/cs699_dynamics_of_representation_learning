import argparse
import logging
import os
import sys

from math import ceil,floor

from train import *
from plot import *
from create_directions import * 
from compute_trajectory import * 
from compute_loss_surface import *


def launch_experiment(args): 
	train_input = ["-D"] 
	create_direction_input = [] 
	compute_trajectory_input = []
	compute_loss_surface_input = [] 
	plot_input = [] 

	# set up the model info
	train_input = train_input + ["--model", args.model]
	create_direction_input = create_direction_input + ["--model", args.model]
	compute_trajectory_input = compute_trajectory_input + ["--model", args.model]
	compute_loss_surface_input = compute_loss_surface_input + ["--model", args.model]
	plot_input = plot_input + ["--model", args.model]


	# set up the augment 
	if args.data_augment:
		train_input = train_input + ["--data_augment"]

	# set up the skip bn bias 
	if args.skip_bn_bias:
		train_input = train_input + ["--skip_bn_bias"]
		create_direction_input = create_direction_input + ["--skip_bn_bias"]
		compute_trajectory_input = compute_trajectory_input + ["--skip_bn_bias"]
		compute_loss_surface_input = compute_loss_surface_input + ["--skip_bn_bias"]
		plot_input = plot_input + ["--skip_bn_bias"]

	# set up the remove skip connection 
	if args.remove_skip_connections:
		train_input = train_input + ["--remove_skip_connections"]
		create_direction_input = create_direction_input + ["--remove_skip_connections"]
		compute_trajectory_input = compute_trajectory_input + ["--remove_skip_connections"]
		compute_loss_surface_input = compute_loss_surface_input + ["--remove_skip_connections"]
		plot_input = plot_input + ["--remove_skip_connections"]

	# set up the device info 
	train_input = train_input + ["--device", args.device]
	compute_loss_surface_input = compute_loss_surface_input + ["--device", args.device]

	# set up the file locations and directories 
	train_result_folder = "results/" + args.exp_name + "/"
	train_input = train_input + ["--result_folder",train_result_folder]

	statefile_folder = train_result_folder + "/ckpt/"
	direction_file = "pca_directions.npz"
	create_direction_input = create_direction_input + ["--statefile_folder",statefile_folder]
	create_direction_input = create_direction_input + ["-r",train_result_folder]
	create_direction_input = create_direction_input + ["--direction_style","pca","--direction_file",direction_file]

	trajectory_folder = train_result_folder + "/trajectories/"
	projection_file = "pca_dir_proj.npz"
	compute_trajectory_input = compute_trajectory_input + ["-r",trajectory_folder]
	compute_trajectory_input = compute_trajectory_input + ["--projection_file",projection_file]
	compute_trajectory_input = compute_trajectory_input + ["-s",statefile_folder]
	compute_trajectory_input = compute_trajectory_input + ["--direction_file", train_result_folder + direction_file]

	surface_folder = train_result_folder + "/loss_surface/"
	target_statefile_loc = statefile_folder + "/120_model.pt"
	surface_file = "pca_dir_loss_surface.npz"

	compute_loss_surface_input = compute_loss_surface_input + ["--result_folder",surface_folder]
	compute_loss_surface_input = compute_loss_surface_input + ["--surface_file",surface_file]
	compute_loss_surface_input = compute_loss_surface_input + ["-s",target_statefile_loc]
	compute_loss_surface_input = compute_loss_surface_input + ["--direction_file", train_result_folder + direction_file]
	compute_loss_surface_input = compute_loss_surface_input + ["--batch_size", "5000"]

	plot_input = plot_input + ["--result_folder", train_result_folder + "/figures/"]
	plot_input = plot_input + ["--trajectory_file", trajectory_folder + projection_file]
	plot_input = plot_input + ["--surface_file", surface_folder + surface_file]
	plot_input = plot_input + ["--plot_prefix", "resnet20_pca_dir"]
	
	train_args = get_train_args(train_input)
	train(train_args) 

	create_direction_args = get_create_direction_args(create_direction_input) 
	create_direction(create_direction_args)

	compute_trajectory_input = compute_trajectory_input + ["--skip_bn_bias"]
	compute_trajectory_args = get_compute_trajectory_args(compute_trajectory_input)
	xcoords,ycoords = compute_trajectory(compute_trajectory_args) 
	
	compute_loss_surface_input = compute_loss_surface_input + ["--skip_bn_bias"]
	compute_loss_surface_input = compute_loss_surface_input + ["--xcoords","51:%.3lf:%.3lf"%(xcoords[0],xcoords[1])]
	compute_loss_surface_input = compute_loss_surface_input + ["--ycoords","51:%.3lf:%.3lf"%(ycoords[0],ycoords[1])]

	compute_loss_surface_args = get_compute_loss_surface_args(compute_loss_surface_input)
	compute_loss_surface(compute_loss_surface_args) 

	plot_args = get_plot_args(plot_input) 
	plot(plot_args)

def get_experiment_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-D", "--debug", action='store_true')
	parser.add_argument("--seed", required=False, type=int, default=0)
	parser.add_argument(
		"--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
	)
	parser.add_argument(
		"--model", required=True, choices=["resnet20", "resnet32", "resnet44", "resnet56"]
	)
	parser.add_argument("--skip_bn_bias", action="store_true", default=False)
	parser.add_argument("--remove_skip_connections", action="store_true", default=False)
	parser.add_argument("--exp_name", "-exp_name", required=True)
	parser.add_argument("--data_augment", action="store_true", default=True) 
	
	return parser.parse_args()

if __name__ == '__main__':
	args = get_experiment_args()
	launch_experiment(args) 

