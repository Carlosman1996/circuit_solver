import pytest
import numpy as np


def test_euler_method_one_dimension(methods_import_one_dimension):
    assert methods_import_one_dimension.method_solver('Euler')[-1][0] == 7.0625


def test_heun_method_one_dimension(methods_import_one_dimension):
    assert methods_import_one_dimension.method_solver('Heun')[-1][0] == 8.9729


def test_runge_kutta_method_one_dimension(methods_import_one_dimension):
    assert methods_import_one_dimension.method_solver('Runge-Kutta')[-1][0] == 9.384


def test_equations_system(methods_import_equations_system):
    euler_last_value = methods_import_equations_system.method_solver('Euler')[-1][0]
    heun_last_value = methods_import_equations_system.method_solver('Heun')[-1][0]
    runge_kutta_last_value = methods_import_equations_system.method_solver('Runge-Kutta')[-1][0]

    assert np.isclose(euler_last_value, heun_last_value, rtol=0.01)
    assert np.isclose(heun_last_value, runge_kutta_last_value, rtol=0.01)
