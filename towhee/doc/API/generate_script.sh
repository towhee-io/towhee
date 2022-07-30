#!/bin/bash
mkdir /towhee/towhee/doc/API/_source;
sphinx-apidoc --tocfile API -f -M -t=./towhee/towhee/doc/API/_templates -e -o ./towhee/towhee/doc/API/_source ./towhee/towhee;
(cd ./towhee/towhee/doc/API && make html);