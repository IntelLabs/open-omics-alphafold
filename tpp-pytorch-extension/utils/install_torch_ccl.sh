#!/bin/bash

###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

GCC_GOOD=`gcc --version | awk '/gcc/ && ($3+0)>=8.3{print "1"}'`
if [ "x$GCC_GOOD" != "x1" ] ; then
  echo "Requires gcc version later than 8.3.0"
  exit 1
fi

pt_version=$(python -c "import torch; print(torch.__version__)" 2> /dev/null)
if [ "x$pt_version" == "x" ] ; then
  echo "Can't find pytorch version, need PyTorch 1.9 or higher..."
  exit 1
fi

branch=$(echo $pt_version | tr "." " " | awk '{print "ccl_torch" $1 "." $2}')

if ! test -d ./torch-ccl ; then
  git clone https://github.com/intel/torch-ccl.git
fi
cd torch-ccl
# workaround to disable linker error for linking to mkl libraries
# export CMAKE_FIND_DEBUG_MODE=ON
export CMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE
git checkout $branch && git submodule sync && git submodule update --init --recursive && CC=gcc CXX=g++ CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ python setup.py install

