#!/bin/bash
mkdir /towhee/towhee/doc/API/_source;
sphinx-apidoc --tocfile API -f -M -t=./towhee/towhee/doc/API/_templates -e -o ./towhee/towhee/doc/API/full_api ./towhee/towhee;
sphinx-apidoc --tocfile API -f -M -t=./towhee/towhee/doc/API/_templates -e -o ./towhee/towhee/doc/API/user_api ./towhee/towhee/functional;
(cd ./towhee/towhee/doc/API && make html);