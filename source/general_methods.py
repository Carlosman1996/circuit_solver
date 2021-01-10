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

    def euler_point_equation(self, point, values):
        return values + self.interval * self.equations_system(point, values)

    def euler_method_solver(self) -> list:
        result = []

        result.append(self.initial_conditions)
        for point in self.points[:-1]:
            result.append(self.euler_point_equation(point, result[-1]))
        return result

    def heun_method_solver(self) -> list:
        result = []

        result.append(self.initial_conditions)
        for index_point, point in enumerate(self.points[:-1]):
            euler_result = self.euler_point_equation(point, result[-1])
            result.append(result[-1] + self.interval / 2 * (self.equations_system(point, result[-1]) + self.equations_system(self.points[index_point + 1], euler_result)))
        return result

    def runge_kutta_method_solver(self) -> list:
        result = []

        result.append(self.initial_conditions)
        for point in self.points[:-1]:
            m1 = self.equations_system(point, result[-1])
            m2 = self.equations_system(point + self.interval / 2, result[-1] + self.interval * m1 / 2)
            m3 = self.equations_system(point + self.interval / 2, result[-1] + self.interval * m2 / 2)
            m4 = self.equations_system(point + self.interval, result[-1] + self.interval * m3)

            result.append(result[-1] + self.interval / 6 * (m1 + 2 * m2 + 2 * m3 + m4))
        return result

    def plotter(self, results):
        fig, ax = plt.subplots(self.number_parameters, 1)

        fig.suptitle('Results')
        plt.xlabel('time (s)')
        plt.ylabel('y axis')

        for model, result in results.items():
            if len(result) > 0:
                if self.number_parameters > 1:
                    for index_parameter in range(0, self.number_parameters):
                        ax[self.number_parameters, 0].set_title(f'Solutions for {index_parameter} element')
                        ax[self.number_parameters, 0].plot(self.points, [specific_result[index_parameter] for specific_result in result],
                                                           label=f'Numerical Solution using {model} model')
                else:
                    ax.set_title(f'Solutions')
                    ax.plot(self.points, result,
                            label=f'Numerical Solution using {model} model')

        plt.legend(loc='upper center', shadow=True, fontsize='medium')
        plt.show()


if __name__ == "__main__":
    def equations_system(amplitude, resistance, frequency, inductance):
        return lambda point, values: np.array([(amplitude * np.sin(2 * np.pi * frequency * point) - values[0] * resistance) / inductance])

    methods = Methods(equations_system=equations_system(amplitude=1,
                                                        resistance=2,
                                                        frequency=100000,
                                                        inductance=50e-6),
                      steps=1000,
                      start_time=0.0001,
                      stop_time=0.0005,
                      initial_conditions=[2])

    results_dictionary = {}

    results_dictionary.setdefault('Euler', methods.euler_method_solver())
    results_dictionary.setdefault('Heun', methods.heun_method_solver())
    results_dictionary.setdefault('Runge-Kutta', methods.runge_kutta_method_solver())

    methods.plotter(results_dictionary)
