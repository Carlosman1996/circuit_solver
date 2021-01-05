import context
import pytest
from general_methods import Methods


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
