#!/bin/bash

# cron does not read env, save it here
env > /root/env
exec "$@"
