CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/yaw/config0.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/yaw/config1.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/yaw/config2.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/yaw/config3.yaml;

CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/pitch/config0.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/pitch/config1.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/pitch/config2.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/pitch/config3.yaml;

CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/thrust/config0.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/thrust/config1.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/thrust/config2.yaml;
CUDA_VISIBLE_DEVICES=6 python ../train.py ../configs/thrust/config3.yaml;
