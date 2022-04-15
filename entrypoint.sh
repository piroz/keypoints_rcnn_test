#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}
USER_NAME=${LOCAL_UNAME:-hoge}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID, UNAME: $USER_NAME"
useradd -u $USER_ID -o -m $USER_NAME
groupmod -g $GROUP_ID $USER_NAME
export HOME=/home/$USER_NAME

chown $USER_ID:$GROUP_ID Pipfile Pipfile.lock entrypoint.sh /app
/usr/sbin/gosu $USER_NAME python3 -m pipenv sync
/usr/sbin/gosu $USER_NAME python3 -m pipenv run pip install git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI

echo "command-line argument:"
echo "$@"

exec /usr/sbin/gosu $USER_NAME "$@"