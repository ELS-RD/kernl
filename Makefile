.PHONY: source_code_format
source_code_format:
	black --line-length 120 --target-version py37 . && \
	isort .

.PHONY: source_code_check_format
source_code_check_format:
	black --check --line-length 120 --target-version py39 . && \
	isort --check-only . && \
	flake8 .


.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build -t kernl .