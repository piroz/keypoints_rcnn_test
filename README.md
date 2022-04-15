# docker enviromnent

install 

- cuda-drivers
- nvidia-container-toolkit
- docker (currently)

# python enviromnent

```
docker build . -t keypointrcnn
```

# run train

## docker gpu (use 10GB vram)

```
nohup docker run --gpus all --rm \
--oom-kill-disable=true \
-e TF_FORCE_GPU_ALLOW_GROWTH=true -e TZ=Asia/Tokyo \
-e LOCAL_UID=$(id -u) -e LOCAL_GID=$(id -g) -e LOCAL_UNAME=$(whoami) \
-v $(pwd):/app \
keypointrcnn python3 -m pipenv run python /app/train.py &
```

## docker cpu

```
nohup docker run --rm \
--oom-kill-disable=true \
-e TZ=Asia/Tokyo \
-e LOCAL_UID=$(id -u) -e LOCAL_GID=$(id -g) -e LOCAL_UNAME=$(whoami) \
-v $(pwd):/app \
keypointrcnn python3 -m pipenv run python /app/train.py &
```

# run predict

## docker gpu

```
nohup docker run --gpus all --rm \
--oom-kill-disable=true \
-e TF_FORCE_GPU_ALLOW_GROWTH=true -e TZ=Asia/Tokyo \
-e LOCAL_UID=$(id -u) -e LOCAL_GID=$(id -g) -e LOCAL_UNAME=$(whoami) \
-v $(pwd):/app \
keypointrcnn python3 -m pipenv run python /app/predict.py &
```

## docker cpu

```
nohup docker run --rm \
--oom-kill-disable=true \
-e TZ=Asia/Tokyo \
-e LOCAL_UID=$(id -u) -e LOCAL_GID=$(id -g) -e LOCAL_UNAME=$(whoami) \
-v $(pwd):/app \
keypointrcnn python3 -m pipenv run python /app/predict.py &
```

# for windows no docker

```
export PIPENV_VENV_IN_PROJECT=1
py -3.8 -m pip install --upgrade pip
py -3.8 -m pip install pipenv
py -3.8 -m pipenv sync
py -3.8 -m pipenv run pip install git+https://github.com/philferriere/cocoapi#subdirectory=PythonAPI
py -3.8 -m pipenv run python train.py
```