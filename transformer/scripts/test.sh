CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/yaw/config0.yaml log_yaw/lightning_logs/version_0/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/yaw/config1.yaml log_yaw/lightning_logs/version_1/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/yaw/config2.yaml log_yaw/lightning_logs/version_2/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/yaw/config3.yaml log_yaw/lightning_logs/version_3/checkpoints;

CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/pitch/config0.yaml log_pitch/lightning_logs/version_0/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/pitch/config1.yaml log_pitch/lightning_logs/version_1/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/pitch/config2.yaml log_pitch/lightning_logs/version_2/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/pitch/config3.yaml log_pitch/lightning_logs/version_3/checkpoints;

CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/thrust/config0.yaml log_thrust/lightning_logs/version_0/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/thrust/config1.yaml log_thrust/lightning_logs/version_1/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/thrust/config2.yaml log_thrust/lightning_logs/version_2/checkpoints;
CUDA_VISIBLE_DEVICES=1 python ../test.py ../configs/thrust/config3.yaml log_thrust/lightning_logs/version_3/checkpoints;
