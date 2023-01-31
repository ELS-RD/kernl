# Whisper optimization

Application of `Kernl` optimizations to Whisper library.  
To run the notebook through shell, use the following command:

```shell
DOCKER_BUILDKIT=1 docker build -t kernl .
docker run --rm -it --gpus all -v $(pwd):/kernl kernl
apt install libsndfile1-dev # used by a Python audio dependency
pip install datasets soundfile librosa jupyter notebook
jupyter nbconvert --execute --clear-output experimental/whisper/speedup.ipynb --log-level=10
```
