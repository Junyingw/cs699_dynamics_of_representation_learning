# # ResNet50, DomainNet, Single Source
# CUDA_VISIBLE_DEVICES=0 python adda.py data/domainnet -d DomainNet -s r -t c -a resnet50 --bottleneck-dim 1024 --epochs 30 -i 2500 -p 500 --seed 0 --log logs/adda/DomainNet_r2c

# # ResNet50, DomainNet, Multi Source
# CUDA_VISIBLE_DEVICES=0 python adda.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 --bottleneck-dim 1024 --epochs 40 -i 5000 -p 500 --seed 0 --log logs/adda/DomainNet_:2c

# Digits
CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s MNIST -t USPS --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool --lr 0.01 -b 128 -i 2500 --scratch --seed 0 --log logs/adda/MNIST2USPS
# CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s USPS -t MNIST --train-resizing 'res.' --val-resizing 'res.' \
#   --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool --lr 0.1 -b 128 -i 2500 --scratch --seed 0 --log logs/adda/USPS2MNIST
# CUDA_VISIBLE_DEVICES=0 python adda.py data/digits -d Digits -s SVHNRGB -t MNISTRGB --train-resizing 'res.' --val-resizing 'res.' \
#   --resize-size 32 --no-hflip --norm-mean 0.5 0.5 0.5 --norm-std 0.5 0.5 0.5 -a dtn --no-pool --trade-off 0.3 --lr 0.03 --lr-d 0.03 -b 128 -i 2500 --scratch --seed 0 --log logs/adda/SVHN2MNIST
