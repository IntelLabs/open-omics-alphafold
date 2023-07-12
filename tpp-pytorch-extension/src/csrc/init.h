/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <string>
#include <vector>

typedef void (*submodule_init_func)(pybind11::module&);

inline std::vector<std::pair<std::string, submodule_init_func>>&
get_submodule_list() {
  static std::vector<std::pair<std::string, submodule_init_func>>
      _submodule_list;
  return _submodule_list;
}

inline int register_submodule(std::string name, submodule_init_func func) {
  auto& _submodule_list = get_submodule_list();
  _submodule_list.push_back(std::make_pair(name, func));
  // printf("Registering %s submodule @ %d\n", name.c_str(),
  // _submodule_list.size()-1);
  return _submodule_list.size();
}

#define REGISTER_SUBMODULE_IMPL(name) \
  static int PySubModule_##name =     \
      register_submodule(#name, PYBIND11_CONCAT(pybind11_init_, name))

#define REGISTER_SUBMODULE(name, variable)                              \
  static void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module&); \
  REGISTER_SUBMODULE_IMPL(name);                                        \
  void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module & variable)
