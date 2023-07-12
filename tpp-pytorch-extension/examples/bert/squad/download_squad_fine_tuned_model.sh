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

#https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main
#mkdir bert-large-uncased-whole-word-masking-finetuned-squad
set -e
mkdir squad_finetuned_checkpoint && cd squad_finetuned_checkpoint
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer_config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt

