import os
import ray
from ray import tune

@ray.remote(num_gpus=1)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


if __name__ == "__main__":

	ray.init()
	print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
	#print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
