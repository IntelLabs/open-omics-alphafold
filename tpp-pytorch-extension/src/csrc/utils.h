/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TPP_UTILS_H_
#define _TPP_UTILS_H_

#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include "bfloat8.h"
#include "vla.h"

#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#define MAX_THREADS 640
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define TPP_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)

typedef at::BFloat16 bfloat16;
typedef at::Half half;
typedef at::BFloat8 bfloat8;

#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define DECL_VLA_PTR_PT(type, name, dims, t) \
  type(*name) dims = (type(*) dims)(pt_get_data_ptr<type>(t))

extern double ifreq; // defined in init.cpp

// Defined in xsmm.cpp
extern thread_local unsigned int* rng_state;
extern thread_local struct drand48_data drng_state; // For non AVX512 version
unsigned int* get_rng_state();

#if 0
template <typename T>
inline T* pt_get_data_ptr(at::Tensor t) {
  return t.data_ptr<T>();
}
#ifndef PYTORCH_SUPPORTS_BFLOAT8
template <>
inline bfloat8* pt_get_data_ptr<bfloat8>(at::Tensor t) {
  // TPP_ASSERT(t.dtype() == at::kByte, "Wrong prec");
  return (bfloat8*)t.data_ptr<uint8_t>();
}
#endif
#endif

template <typename T>
inline std::string get_class_name() {
  auto cname = abi::__cxa_demangle(typeid(T).name(), 0, 0, NULL);
  std::string name(cname);
  free(cname);
  return name;
}

#ifdef __x86_64__
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#elif defined(__aarch64__)
static __inline__ unsigned long long rdtsc(void) {
  unsigned long long val;

  /*
   * According to ARM DDI 0487F.c, from Armv8.0 to Armv8.5 inclusive, the
   * system counter is at least 56 bits wide; from Armv8.6, the counter
   * must be 64 bits wide.  So the system counter could be less than 64
   * bits wide and it is attributed with the flag 'cap_user_time_short'
   * is true.
   */
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));

  return val;
}
#else
#error "Unsupported architecture for rdtsc"
#endif
inline double getFreq() {
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

inline int guess_mpi_rank() {
  const char* env_names[] = {
      "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};
  static int guessed_rank = -1;
  if (guessed_rank >= 0)
    return guessed_rank;
  for (int i = 0; i < 4; i++) {
    if (getenv(env_names[i]) != NULL) {
      int r = atoi(getenv(env_names[i]));
      if (r >= 0) {
        printf("My guessed rank = %d\n", r);
        guessed_rank = r;
        return guessed_rank;
      }
    }
  }
  guessed_rank = 0;
  return guessed_rank;
}

template <int maxlen>
class SafePrint {
 public:
  SafePrint() {}
  template <typename... Types>
  int operator()(Types... vars) {
    if (len < maxlen) {
      int l = snprintf(&buf[len], 2 * maxlen - len, vars...);
      len += l;
      if (len > maxlen) {
        print();
      }
      return l;
    }
    return 0;
  }
  void print() {
    printf("%s", buf);
    len = 0;
  }

 private:
  char buf[2 * maxlen];
  int len = 0;
};

#endif //_TPP_UTILS_H_
