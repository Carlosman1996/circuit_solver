from dataclasses import dataclass
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Methods:
    function: Callable[[float], float]
    steps: float
    start_time: float
    stop_time: float
    initial_condition: float

    def __post_init__(self) -> None:
        self.interval = (self.stop_time - self.start_time) / self.steps
        self.number_points = self.steps + 1
        self.points = np.linspace(start=self.start_time,
                                  stop=self.stop_time, num=self.number_points)

    def euler_point_equation(self, point, value):
        return value + self.interval * self.function(point, value)

    def euler_method_solver(self) -> list:
        result = np.zeros(self.number_points)

        result[0] = self.initial_condition
        for index_point, point in enumerate(self.points[:-1]):
            result[index_point + 1] = self.euler_point_equation(point, result[index_point])
        return result

    def heun_method_solver(self) -> list:
        result = np.zeros(self.number_points)

        result[0] = self.initial_condition
        for index_point, point in enumerate(self.points[:-1]):
            euler_result = self.euler_point_equation(point, result[index_point])
            result[index_point + 1] = result[index_point] + self.interval / 2 * (self.function(point, result[index_point]) + self.function(self.points[index_point + 1], euler_result))
        return result

    def runge_kutta_method_solver(self) -> list:
        result = np.zeros(self.number_points)

        result[0] = self.initial_condition
        for index_point, point in enumerate(self.points[:-1]):
            m1 = self.function(point, result[index_point])
            m2 = self.function(point + self.interval / 2, result[index_point] + self.interval * m1 / 2)
            m3 = self.function(point + self.interval / 2, result[index_point] + self.interval * m2 / 2)
            m4 = self.function(point + self.interval, result[index_point] + self.interval * m3)

            result[index_point + 1] = result[index_point] + self.interval / 6 * (m1 + 2 * m2 + 2 * m3 + m4)
        return result

    def plotter(self, results):
        fig, ax = plt.subplots()

        plt.title('Numerical models comparison')
        plt.xlabel('time (s)')
        plt.ylabel('y axis')

        for model, result in results.items():
            if len(result) > 0:
                ax.plot(self.points, result,
                        label=f'Numerical Solution using {model} model')

        ax.legend(loc='upper center', shadow=True, fontsize='medium')
        plt.show()


if __name__ == "__main__":
    def function(amplitude, resistance, frequency, inductance):
        return lambda point, y: (amplitude * np.sin(2 * np.pi * frequency * point) - y * resistance) / inductance

    methods = Methods(function=function(amplitude=1,
                                        resistance=2,
                                        frequency=100000,
                                        inductance=50e-6),
                      steps=1000,
                      start_time=0.0001,
                      stop_time=0.0005,
                      initial_condition=2)

    results_dictionary = {}

    results_dictionary.setdefault('Euler', methods.euler_method_solver())
    results_dictionary.setdefault('Heun', methods.heun_method_solver())
    results_dictionary.setdefault('Runge-Kutta', methods.runge_kutta_method_solver())

    methods.plotter(results_dictionary)
