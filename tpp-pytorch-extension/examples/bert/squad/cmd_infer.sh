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

NUMA_ARGS=""
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then
      if command -v numactl >& /dev/null ; then
        NUMA_ARGS="numactl -m 0 "
      fi
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
  fi
  NUM_RANKS=$PMI_SIZE
else
  if command -v numactl >& /dev/null ; then
    NUMA_ARGS="numactl -m 0 "
  fi
  NUM_RANKS=1
fi

if [ "x$1" == "x-gdb" ] ; then
GDB_ARGS="gdb --args "
shift
else
GDB_ARGS=""
fi

# set dataset and model_path
if test -z $dataset || ! test -d $dataset ; then
  if test -d ./SQUAD1 ; then
    dataset=./SQUAD1
  else
    echo "Unable to find dataset path"
    exit 1
  fi
fi

if test -z $model_path || ! test -d $model_path ; then
  if test -d ./squad_finetuned_checkpoint ; then
    model_path=./squad_finetuned_checkpoint
  else
    echo "Unable to find model path"
    exit 1
  fi
fi

$NUMA_RAGS $GDB_ARGS python -u run_squad.py \
  --model_type bert \
  --model_name_or_path $model_path \
  --do_eval \
  --do_lower_case \
  --predict_file $dataset/dev-v1.1.json \
  --per_gpu_eval_batch_size 24 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  $@

