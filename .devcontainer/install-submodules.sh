#!/usr/bin/env bash

set -e

cd extern

# install pybind11
git submodule add -b stable https://github.com/pybind/pybind11.git pybind11
git submodule update --init

# install box2d
git submodule add https://github.com/erincatto/box2d/releases/latest box2d
git submodule update --init