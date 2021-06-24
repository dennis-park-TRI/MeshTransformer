PROJECT = metro
WORKSPACE = /workspace/$(PROJECT)
DOCKER_IMAGE = $(PROJECT):latest

BASE_DOCKER_IMAGE ?= nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-v /data:/data \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /mnt/fsx:/mnt/fsx \
	-v /root/.ssh:/root/.ssh \
	-v $(HOME)/.ouroboros:/root/.ouroboros \
	--shm-size=1G \
	--ipc=host \
	--network=host \
	--privileged \
	-e DATASETS_ROOT=/mnt/fsx/datasets/

DOCKER_BUILD_ARGS = \
	--build-arg WORKSPACE=$(WORKSPACE) \
	--build-arg AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_DEFAULT_REGION \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \
	--build-arg BASE_DOCKER_IMAGE=$(BASE_DOCKER_IMAGE)


docker-build:
	docker build \
	$(DOCKER_BUILD_ARGS) \
	-f ./Dockerfile \
	-t $(DOCKER_IMAGE) .

docker-dev:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	-v $(PWD):$(WORKSPACE) \
	$(DOCKER_IMAGE) bash

docker-start:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

dist-run:
	nvidia-docker run --name $(PROJECT) --rm \
		-e DISPLAY=${DISPLAY} \
		-v ~/.torch:/root/.torch \
		${DOCKER_OPTS} \
		-v $(PWD):$(WORKSPACE) \
		${DOCKER_IMAGE} \
		${COMMAND}

docker-run:
	nvidia-docker run --name $(PROJECT) --rm \
		-e DISPLAY=${DISPLAY} \
		-v ~/.torch:/root/.torch \
		${DOCKER_OPTS} \
		${DOCKER_IMAGE} \
		${COMMAND}

clean:
	find . -name '"*.pyc' | xargs rm -f && \
	find . -name '__pycache__' | xargs rm -rf
