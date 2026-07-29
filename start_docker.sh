#!/bin/bash
docker run --rm -it -v "$PWD":/workspace -w /workspace stack-dev /bin/bash
