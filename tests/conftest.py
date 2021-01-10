import pytest
import numpy as np
import context
from general_methods import Methods


@pytest.fixture
def methods_import_one_dimension():
    def function():
        return lambda point, values: 2 * values[0] - 2 * point - 1
    return Methods(equations_system=function(),
                   steps=4,
                   start_time=0,
                   stop_time=1,
                   initial_conditions=[2])


@pytest.fixture
def methods_import_equations_system():
    def function():
        return lambda point, values: np.array([-4 * values[1] + np.cos(point), values[0]])
    return Methods(equations_system=function(),
                   steps=5,
                   start_time=0,
                   stop_time=0.5,
                   initial_conditions=[0, 0])
