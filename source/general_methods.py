from dataclasses import dataclass
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Methods:
    equations_system: Callable[[float], float]
    steps: float
    start_time: float
    stop_time: float
    initial_conditions: float

    def __post_init__(self) -> None:
        self.number_parameters = len(self.initial_conditions)
        self.interval = (self.stop_time - self.start_time) / self.steps
        self.number_points = self.steps + 1
        self.points = np.linspace(start=self.start_time,
                                  stop=self.stop_time, num=self.number_points)

    def solve(self, values, function_values):
        point_result = np.zeros(self.number_parameters)

        for index_value in range((self.number_parameters - 1), -1, -1):
            point_result[index_value] = values[index_value] + self.interval * function_values[index_value]
            if index_value > 1:
                function_values[index_value - 1] = point_result[index_value]
        return point_result

    def euler_point_equation(self, point, values):
        function_values = self.equations_system(point, values)
        return self.solve(values, function_values)

    def heun_point_equation(self, point, next_point, values):
        euler_result = self.euler_point_equation(point, values)
        function_values = 1 / 2 * (self.equations_system(point, values) + self.equations_system(next_point, euler_result))
        return self.solve(values, function_values)

    def runge_kutta_point_equation(self, point, values) -> list:
        m1 = self.equations_system(point, values)
        m2 = self.equations_system(point + self.interval / 2, values + self.interval * m1 / 2)
        m3 = self.equations_system(point + self.interval / 2, values + self.interval * m2 / 2)
        m4 = self.equations_system(point + self.interval, values + self.interval * m3)
        function_values = 1 / 6 * (m1 + 2 * m2 + 2 * m3 + m4)
        return self.solve(values, function_values)

    def method_solver(self, method) -> list:
        result = []

        result.append(self.initial_conditions)
        for index_point, point in enumerate(self.points[:-1]):
            if method == 'Euler':
                result.append(self.euler_point_equation(point, result[-1]))
            elif method == 'Heun':
                result.append(self.heun_point_equation(point, self.points[index_point + 1], result[-1]))
            elif method == 'Runge-Kutta':
                result.append(self.runge_kutta_point_equation(point, result[-1]))
            else:
                raise NotImplementedError
        return result

    def plotter(self, results):
        fig, ax = plt.subplots(self.number_parameters)

        fig.suptitle('Results')

        for model, result in results.items():
            if len(result) > 0:
                for index_parameter in range(0, self.number_parameters):
                    ax[index_parameter].set_xlabel('time (s)')
                    ax[index_parameter].set_ylabel('y axis')
                    ax[index_parameter].set_title(f'Solutions for element: {index_parameter}')
                    ax[index_parameter].plot(self.points, [specific_result[index_parameter] for specific_result in result],
                                             label=f'Numerical Solution using {model} model')
                    ax[index_parameter].legend(loc='upper center', shadow=True, fontsize='medium')

        plt.show()


if __name__ == "__main__":
    def equations_system(amplitude, resistance, inductance, capacitance, radial_frequency):
        return lambda point, values: np.array([values[1], (amplitude * radial_frequency * np.cos(radial_frequency * point) - values[0] / capacitance - resistance * values[1]) / inductance])

    methods = Methods(equations_system=equations_system(amplitude=5,
                                                        resistance=2,
                                                        inductance=1,
                                                        capacitance=1,
                                                        radial_frequency=1),
                      steps=1000,
                      start_time=0,
                      stop_time=20,
                      initial_conditions=[0, 100])

    results_dictionary = {}

    results_dictionary.setdefault('Euler', methods.method_solver('Euler'))
    results_dictionary.setdefault('Heun', methods.method_solver('Heun'))
    results_dictionary.setdefault('Runge-Kutta', methods.method_solver('Runge-Kutta'))

    methods.plotter(results_dictionary)
