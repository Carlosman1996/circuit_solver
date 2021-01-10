import pytest
import numpy as np


def simplify(array):
    return list(np.around(np.sum(array, axis=1), 4))


def test_euler_method_one_dimension(methods_import_one_dimension):
    assert simplify(methods_import_one_dimension.euler_method_solver()) == [2, 2.75, 3.75, 5.125, 7.0625]


def test_heun_method_one_dimension(methods_import_one_dimension):
    assert simplify(methods_import_one_dimension.heun_method_solver()) == [2.0, 2.875, 4.1406, 6.041, 8.9729]


def test_runge_kutta_method_one_dimension(methods_import_one_dimension):
    assert simplify(methods_import_one_dimension.runge_kutta_method_solver()) == [2.0, 2.8984, 4.2173, 6.2294, 9.384]


def test_euler_method_equations_system(methods_import_equations_system):
    assert simplify(methods_import_equations_system.euler_method_solver()) == simplify([[0, 0], [0.1, 0], [0.1995, 0.01], [0.2935, 0.03], [0.3771, 0.0593], [0.4455, 0.097]])


def test_heun_method_equations_system(methods_import_equations_system):
    assert simplify(methods_import_equations_system.heun_method_solver()) == simplify([[0, 0], [0.0998, 0.005], [0.1945, 0.01985], [0.2795, 0.0438], [0.3498, 0.076], [0.4028, 0.1138]])


def test_runge_kutta_method_equations_system(methods_import_equations_system):
    assert simplify(methods_import_equations_system.runge_kutta_method_solver()) == simplify([[0, 0], [0.0991, 0.005], [0.1934, 0.0197], [0.2779, 0.0434], [0.3484, 0.0748], [0.4012, 0.1124]])
