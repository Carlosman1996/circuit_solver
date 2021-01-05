import context
import pytest
from general_methods import Methods
import numpy as np


@pytest.fixture
def methods_import():
    def function():
        return lambda point, y: 2 * y - 2 * point - 1
    return Methods(function=function(),
                   steps=4,
                   start_time=0,
                   stop_time=1,
                   initial_condition=2)


def test_euler_method(methods_import):
    assert list(methods_import.euler_method_solver()) == [2, 2.75, 3.75, 5.125, 7.0625]


def test_heun_method(methods_import):
    assert list(np.around(methods_import.heun_method_solver(), 4)) == [2.0, 2.875, 4.1406, 6.041, 8.9729]


def test_runge_kutta_method(methods_import):
    assert list(np.around(methods_import.runge_kutta_method_solver(), 4)) == [2.0, 2.8984, 4.2173, 6.2294, 9.384]
