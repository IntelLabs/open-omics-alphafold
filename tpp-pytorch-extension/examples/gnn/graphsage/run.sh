#!/bin/bash

###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Sasikanth Avancha (Intel Corp.)                                     #
###############################################################################

CORES_PER_SOCKET=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
DATALOADER_WORKER_COUNT=0
if [ $CORES_PER_SOCKET -gt 16 ] ; then 
  if [ $CORES_PER_SOCKET -le 22 ] ; then DATALOADER_WORKER_COUNT=1 ; fi
fi

if [ $CORES_PER_SOCKET -gt 22 ] ; then
  if [ $CORES_PER_SOCKET -lt 28 ] ; then DATALOADER_WORKER_COUNT=2 ; fi 
fi

if [ $CORES_PER_SOCKET -eq 28 ] ; then DATALOADER_WORKER_COUNT=4 ; fi
if [ $CORES_PER_SOCKET -eq 36 ] ; then DATALOADER_WORKER_COUNT=4 ; fi
if [ $CORES_PER_SOCKET -ge 40 ] ; then DATALOADER_WORKER_COUNT=8 ; fi

export OMP_NUM_THREADS=$(( CORES_PER_SOCKET - DATALOADER_WORKER_COUNT ))
unset KMP_AFFINITY
unset KMP_BLOCKTIME

ulimit -u `ulimit -Hu`
ulimit -n `ulimit -Hn`

if [ "x$2" == "x" ] ; then 
  if [ $DATALOADER_WORKER_COUNT -eq 0 ] ; then
    python -u main.py --num-workers $DATALOADER_WORKER_COUNT --fan-out 15,10,5 --dataset $1
  else
    python -u main.py --num-workers $DATALOADER_WORKER_COUNT --cpu-worker-aff --fan-out 15,10,5 --dataset $1
  fi
else
  if [ "x$3" == "x" ] ; then
    if [ $DATALOADER_WORKER_COUNT -eq 0 ] ; then
      python -u main.py --num-workers $DATALOADER_WORKER_COUNT --fan-out 15,10,5 --dataset $1 $2
    else
      python -u main.py --num-workers $DATALOADER_WORKER_COUNT --cpu-worker-aff --fan-out 15,10,5 --dataset $1 $2
    fi
  else
    if [ $DATALOADER_WORKER_COUNT -eq 0 ] ; then
      python -u main.py --num-workers $DATALOADER_WORKER_COUNT --fan-out 15,10,5 --dataset $1 $2 $3
    else
      python -u main.py --num-workers $DATALOADER_WORKER_COUNT --cpu-worker-aff --fan-out 15,10,5 --dataset $1 $2 $3
    fi
  fi
fi
