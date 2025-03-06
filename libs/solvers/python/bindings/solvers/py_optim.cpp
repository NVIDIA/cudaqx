/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/solvers/observe_gradient.h"
#include "cudaq/solvers/optimizer.h"

#include "bindings/utils/type_casters.h"
#include "cuda-qx/core/kwargs_utils.h"

namespace py = pybind11;

namespace cudaq::optim {

void bindOptim(py::module &mod) {

  auto optim = mod.def_submodule("optim");
  py::class_<optimizable_function>(optim, "OptimizableFunction")
      .def(py::init<>(), R"docstring(
        Default constructor for OptimizableFunction.
    )docstring")
      .def(py::init<const optimizable_function &>(), R"docstring(
        Copy constructor for OptimizableFunction.

        Args:
            other (OptimizableFunction): The OptimizableFunction to copy.
    )docstring")
      .def(
          "__call__",
          [](const optimizable_function &self, const std::vector<double> &x) {
            std::vector<double> dx;
            return self(x, dx);
          },
          R"docstring(
        Evaluate the function without returning gradients.

        Args:
            x (List[float]): Input vector.

        Returns:
            float: The function value at x.
    )docstring")
      .def(
          "__call__",
          [](const optimizable_function &self, const std::vector<double> &x,
             std::vector<double> &dx) { return self(x, dx); },
          R"docstring(
        Evaluate the function and compute gradients.

        Args:
            x (List[float]): Input vector.
            dx (List[float]): Output vector to store gradients.

        Returns:
            float: The function value at x.
    )docstring")
      .def("provides_gradients", &optimizable_function::providesGradients,
           R"docstring(
        Check if the function provides gradient information.

        Returns:
            bool: True if the function can compute gradients, False otherwise.
    )docstring");
  optim.def(
      "optimize",
      [](const py::function &function, std::vector<double> xInit,
         std::string method, py::kwargs options) {
        heterogeneous_map optOptions;
        optOptions.insert("initial_parameters", xInit);

        if (!cudaq::optim::optimizer::is_registered(method))
          throw std::runtime_error(
              method + " is not a valid, registered cudaq-x optimizer.");

        auto opt = cudaq::optim::optimizer::get(method);
        auto result = opt->optimize(
            xInit.size(),
            [&](std::vector<double> x, std::vector<double> &grad) {
              // Call the function.
              auto ret = function(x);
              // Does it return a tuple?
              auto isTupleReturn = py::isinstance<py::tuple>(ret);
              // If we don't need gradients, and it does, just grab the value
              // and return.
              if (!opt->requiresGradients() && isTupleReturn)
                return ret.cast<py::tuple>()[0].cast<double>();
              // If we dont need gradients and it doesn't return tuple, then
              // just pass what we got.
              if (!opt->requiresGradients() && !isTupleReturn)
                return ret.cast<double>();

              // Throw an error if we need gradients and they weren't provided.
              if (opt->requiresGradients() && !isTupleReturn)
                throw std::runtime_error(
                    "Invalid return type on objective function, must return "
                    "(float,list[float]) for gradient-based optimizers");

              // If here, we require gradients, and the signature is right.
              auto tuple = ret.cast<py::tuple>();
              auto val = tuple[0];
              auto gradIn = tuple[1].cast<py::list>();
              for (std::size_t i = 0; i < gradIn.size(); i++)
                grad[i] = gradIn[i].cast<double>();

              return val.cast<double>();
            },
            optOptions);

        return result;
      },
      py::arg("function"), py::arg("initial_parameters"),
      py::arg("method") = "cobyla", R"#(
Optimize a given objective function using various optimization methods.

This function performs optimization on a user-provided objective function
using the specified optimization method. It supports both gradient-based
and gradient-free optimization algorithms.

Parameters:
-----------
function : callable
    The objective function to be minimized. It should take a list of parameters
    as input and return either:
    - A single float value (for gradient-free methods)
    - A tuple (float, list[float]) where the first element is the function value and the second is the gradient (for gradient-based methods)
initial_parameters : list[float]
    Initial guess for the parameters to be optimized.
method : str, optional
    The optimization method to use. Default is 'cobyla'.
    Must be a valid, registered cudaq-x optimizer.
options : dict
    Additional options for the optimizer. These are method-specific.

Returns:
--------
OptimizationResult
    An object containing the results of the optimization process.

Raises:
-------
RuntimeError
    If an invalid optimization method is specified or if the objective function
    returns an incorrect format for gradient-based optimizers.

Examples:
---------
>>> def objective(x):
...     return sum([xi**2 for xi in x]), [2*xi for xi in x]
>>> result = optimize(objective, [1.0, 2.0, 3.0], method='l-bfgs-b')
>>> print(result.optimal_parameters)
[0.0, 0.0, 0.0]

>>> def simple_objective(x):
...     return sum([xi**2 for xi in x])
>>> result = optimize(simple_objective, [1.0, 2.0, 3.0], method='cobyla')
>>> print(result.optimal_value)
0.0

Notes:
------
- The function automatically detects whether the optimization method requires
  gradients and expects the objective function to return the appropriate format.
- For gradient-based methods, the objective function must return a tuple of
  (value, gradient).
- For gradient-free methods, the objective function should return only the value.
- The optimization process uses the cudaq-x backend, which must be properly
  set up and have the specified optimization method registered.
)#");
}

} // namespace cudaq::optim
