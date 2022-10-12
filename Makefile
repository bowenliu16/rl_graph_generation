.PHONY: build run_dev_gpu run_dev_no_gpu

base_img_with_tag := pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel 
built_img := gcpn
built_tag := dev
local_dir := $(shell pwd)

gpu_args := -e USER_ID=$(shell id -u) \
	-e USER_GP=$(shell id -g) \
	-e USERNAME=$(shell id -un) \
	--env NVIDIA_VISIBLE_DEVICES=all \
	--env NVIDIA_VISIBLE_CAPABILITIES=all \
	--env="QT_X11_NO_MITSHM=1" \
	--env="XAUTHORITY=$(XAUTH)" \
	--volume $(XAUTH):$(XAUTH) \
	--volume /tmp/.X11-unix:/tmp/.X11-unix \

no_gpu_args := -e USER_ID=$(shell id -u) \
	-e USER_GP=$(shell id -g) \
	-e USERNAME=$(shell id -un) \
	--env="QT_X11_NO_MITSHM=1" \
	--env="XAUTHORITY=$(XAUTH)" \
	--volume $(XAUTH):$(XAUTH) \
	--volume /tmp/.X11-unix:/tmp/.X11-unix \

gpu_container := --gpus all -it --init $(gpu_args)
no_gpu_container := -it --init $(no_gpu_args)

build:
	@docker build --build-arg base_img_with_tag=$(base_img_with_tag) . -f Dockerfile --network=host --tag ${built_img}:${built_tag}


run_dev_gpu:
	@docker run -d --net=host $(gpu_container) -v $(local_dir):/usr/src/app/rl_graph_generation $(built_img):$(built_tag)

run_dev_no:
	@docker run -d --net=host $(no_gpu_container) -v $(local_dir):/usr/src/app/rl_graph_generation ${built_img}:${built_tag}
