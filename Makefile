.PHONY: source_code_check_format
source_code_check_format:
	black --check --line-length 120 --target-version py39 . && \
	isort --check-only . && \
	flake8 .

.PHONY: source_code_format
source_code_format:
	black --line-length 120 --target-version py39 . && \
	isort .
	$(MAKE) source_code_check_format

.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build -t kernl .

.PHONY: docker_run
docker_run:
	docker run --rm -it --gpus all -v $(pwd):/kernl kernl
