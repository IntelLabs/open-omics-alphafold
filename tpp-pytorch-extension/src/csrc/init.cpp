/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include "init.h"
#include "timing.h"
#include "utils.h"

#ifdef _OPENMP
#pragma message "Using OpenMP"
#endif

double ifreq = 1.0 / getFreq();

PassType globalPass = OTH;
REGISTER_SCOPE(other, "other");
REGISTER_SCOPE(w_vnni, "w_vnni");
REGISTER_SCOPE(w_xpose, "w_xpose");
REGISTER_SCOPE(a_xpose, "a_xpose");
REGISTER_SCOPE(a_vnni, "a_vnni");
REGISTER_SCOPE(zero, "zero");
REGISTER_SCOPE(pad_act, "pad_act");
REGISTER_SCOPE(unpad_act, "unpad_act");

int globalScope = 0;

void reset_debug_timers() {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (auto& scope : get_pass_list()) {
      if (scope.master_timer == 0.0)
        continue;
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope.detailed_timers[tid][t] = 0.0;
      }
      scope.flops[tid][0] = 0;
    }
    for (auto& scope : get_scope_list()) {
      if (scope.master_timer == 0.0)
        continue;
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope.detailed_timers[tid][t] = 0.0;
      }
      scope.flops[tid][0] = 0;
    }
  }
  for (auto& scope : get_pass_list()) {
    if (scope.master_timer == 0.0)
      continue;
    scope.master_timer = 0.0;
  }
  for (auto& scope : get_scope_list()) {
    if (scope.master_timer == 0.0)
      continue;
    scope.master_timer = 0.0;
  }
}

void print_debug_timers(int tid) {
  int my_rank = guess_mpi_rank();
  if (my_rank != 0)
    return;
  int max_threads = omp_get_max_threads();
  constexpr int maxlen = 10000;
  SafePrint<maxlen> printf;
  // printf("%-20s", "####");
  printf("### ##: %-11s: ", "#KEY#");
  for (int t = 0; t < LAST_TIMER; t++) {
    printf(" %7s", DebugTimerName(t));
  }
  printf(
      " %8s  %8s  %8s (%4s) %6s\n",
      "Total",
      "MTotal",
      "TotalGFS",
      "IMBL",
      "TF/s");
  for (int i = 0; i < max_threads; i++) {
    if (tid == -1 || tid == i) {
      auto print_scope = [&](const Scope& scope) {
        if (scope.master_timer == 0.0)
          return;
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, scope.name.c_str());
        for (int t = 0; t < LAST_TIMER; t++) {
          printf(" %7.1f", scope.detailed_timers[i][t] * 1e3);
          total += scope.detailed_timers[i][t];
        }
        long t_flops = 0;
        for (int f = 0; f < max_threads; f++)
          t_flops += scope.flops[f][0];
        if (t_flops > 0.0) {
          printf(
              " %8.1f  %8.1f  %8.3f (%4.2f) %6.3f\n",
              total * 1e3,
              scope.master_timer * 1e3,
              t_flops * 1e-9,
              t_flops * 100.0 / (scope.flops[i][0] * max_threads),
              t_flops * 1e-12 / scope.detailed_timers[i][BRGEMM]);
        } else {
          printf(" %8.1f  %8.1f\n", total * 1e3, scope.master_timer * 1e3);
        }
      };
      for (auto& scope : get_pass_list())
        print_scope(scope);
      for (auto& scope : get_scope_list())
        print_scope(scope);
    }
  }
  printf.print();
}

void print_debug_thread_imbalance() {
  int my_rank = guess_mpi_rank();
  if (my_rank != 0)
    return;
  int max_threads = omp_get_max_threads();
  constexpr int maxlen = 10000;
  SafePrint<maxlen> printf;
  // printf("%-20s", "####");
  printf("%-11s: ", "#KEY#");
  printf("TID %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
  printf("MIN %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
  printf("MAX %7s %7s %7s  ", DebugTimerName(0), "ELTW", "Total");
  printf(
      " %8s  %9s (%5s) %6s   %9s %9s %9s\n",
      "MTotal",
      "GF_Total",
      "IMBL",
      "TF/s",
      "GF_T0",
      "GF_Tmin",
      "GF_Tmax");
  auto print_scope = [&](const Scope& scope) {
    if (scope.master_timer == 0.0)
      return;
    double total_0 = 0.0;
    for (int t = 0; t < LAST_TIMER; t++) {
      total_0 += scope.detailed_timers[0][t];
    }
    double total_min = total_0;
    double total_max = total_0;
    int total_imin = 0;
    int total_imax = 0;
    for (int i = 1; i < max_threads; i++) {
      double total = 0.0;
      for (int t = 0; t < LAST_TIMER; t++) {
        total += scope.detailed_timers[i][t];
      }
      if (total < total_min) {
        total_imin = i;
        total_min = total;
      }
      if (total > total_max) {
        total_imax = i;
        total_max = total;
      }
    }
    printf("%-11s: ", scope.name.c_str());
    printf(
        "T%02d %7.1f %7.1f %7.1f  ",
        0,
        scope.detailed_timers[0][0] * 1e3,
        (total_0 - scope.detailed_timers[0][0]) * 1e3,
        total_0 * 1e3);
    printf(
        "T%02d %7.1f %7.1f %7.1f  ",
        total_imin,
        scope.detailed_timers[total_imin][0] * 1e3,
        (total_min - scope.detailed_timers[total_imin][0]) * 1e3,
        total_min * 1e3);
    printf(
        "T%02d %7.1f %7.1f %7.1f  ",
        total_imax,
        scope.detailed_timers[total_imax][0] * 1e3,
        (total_max - scope.detailed_timers[total_imax][0]) * 1e3,
        total_max * 1e3);
    long t_flops = 0;
    for (int f = 0; f < max_threads; f++)
      t_flops += scope.flops[f][0];
    if (t_flops > 0.0) {
      printf(
          " %8.1f  %9.3f (%5.2f) %6.3f   %9.3f %9.3f %9.3f\n",
          scope.master_timer * 1e3,
          t_flops * 1e-9,
          t_flops * 100.0 / (scope.flops[0][0] * max_threads),
          t_flops * 1e-12 / scope.detailed_timers[0][BRGEMM],
          scope.flops[0][0] * 1e-9,
          scope.flops[total_imin][0] * 1e-9,
          scope.flops[total_imax][0] * 1e-9);
    } else {
      printf(" %8.1f\n", scope.master_timer * 1e3);
    }
  };
  for (auto& scope : get_pass_list())
    print_scope(scope);
  for (auto& scope : get_scope_list())
    print_scope(scope);
  printf.print();
}

static void init_submodules(pybind11::module& m) {
  auto& _submodule_list = get_submodule_list();
  for (auto& p : _submodule_list) {
    auto sm = m.def_submodule(p.first.c_str());
    auto module = py::handle(sm).cast<py::module>();
    p.second(module);
  }
}

// PYBIND11_MODULE(TORCH_MODULE_NAME, m) {
PYBIND11_MODULE(_C, m) {
  init_submodules(m);
  m.def("print_debug_timers", &print_debug_timers, "print_debug_timers");
  m.def(
      "print_debug_thread_imbalance",
      &print_debug_thread_imbalance,
      "print_debug_thread_imbalance");
  m.def("reset_debug_timers", &reset_debug_timers, "reset_debug_timers");
};
