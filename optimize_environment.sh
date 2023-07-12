unset LD_PRELOAD
f_iomp=$CONDA_PREFIX/lib/libiomp5.so
f_malloc=$CONDA_PREFIX/lib/libjemalloc.so
if [ -f ${f_iomp} ]; do
  export LD_PRELOAD=${f_iomp}
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_BLOCKTIME=0
  export KMP_SETTINGS=0
fi
if [ -f ${f_malloc} ]; do
  export LD_PRELOAD=${f_malloc}:$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
fi

export TF_ENABLE_ONEDNN_OPTS=1
export USE_OPENMP=1
export USE_AVX512=1
export IPEX_ONEDNN_LAYOUT=1
export PYTORCH_TENSOREXPR=0
