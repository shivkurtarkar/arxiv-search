# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: arXiv Search App 
# help:



IMAGE_NAME:=redis-vector-db
IMAGE_VER:=v1
IMAGE_FULL_NAME:=${IMAGE_NAME}:${IMAGE_VER}
REGISTRY:=shivamkurtarkar

# help: help                   - display this makefiles help information
.PHONY: help
help:
	@grep "^# help:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help:
# help: build-push-redis-image - build redis image and push int repository
build-push-redis-image:
	docker build -t ${IMAGE_FULL_NAME} -f redis-vector-db/Dockerfile redis-vector-db
	docker tag ${IMAGE_FULL_NAME} ${REGISTRY}/${IMAGE_FULL_NAME}
	docker push ${REGISTRY}/${IMAGE_FULL_NAME}

# help:
# help: