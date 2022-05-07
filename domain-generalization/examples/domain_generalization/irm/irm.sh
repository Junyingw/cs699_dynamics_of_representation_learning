# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python irm.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 -i 4000 --anneal-iters 4000 --lr 0.005 --seed 0 --log logs/irm/DomainNet_c
