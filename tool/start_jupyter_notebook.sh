#!/usr/bin/env bash

if !which pipenv &> /dev/null; then
    print "pipenv not found, please install pipenv first." >&2
fi

logfile="/tmp/jupyter_notebook-carnd_all-$(date +%Y%m%dT%H%M%S).log"
project_root=$(cd $(dirname $0)/..; pwd)
pipenv run jupyter notebook $project_root \
    --config $project_root/config/jupyter_notebook_config.py &> $logfile &

