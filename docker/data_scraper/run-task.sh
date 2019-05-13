#!/bin/bash

# import env vars that were written in entrypoint
env - `cat /root/env` $@
