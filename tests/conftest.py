import pytest
import numpy as np
import context
from general_methods import Methods


@pytest.fixture
def methods_import_one_dimension():
    def function():
        return lambda point, values: np.array([2 * values[0] - 2 * point - 1])
    return Methods(equations_system=function(),
                   steps=4,
                   start_time=0,
                   stop_time=1,
                   initial_conditions=[2])


@pytest.fixture
def methods_import_equations_system():
    def equations_system(amplitude, resistance, inductance, capacitance, radial_frequency):
        return lambda point, values: np.array([values[1], (amplitude * radial_frequency * np.cos(radial_frequency * point) - values[0] / capacitance - resistance * values[1]) / inductance])
    return Methods(equations_system=equations_system(amplitude=5,
                                                     resistance=2,
                                                     inductance=1,
                                                     capacitance=1,
                                                     radial_frequency=1),
                   steps=1000,
                   start_time=0,
                   stop_time=20,
                   initial_conditions=[0, 100])
