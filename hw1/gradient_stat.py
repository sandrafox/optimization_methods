from gradient_descent import linear_step_chooser, number_of_iters, save, draw_function, backtracking, gradient_descent
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from one_dimesional_search import fib, dichotomy_method, golden_ratio_method


def sum_squares(args):
    return np.sum(args ** 2)

def sum_squares_grad(args):
    return 2 * args


def rosenbrock(arg):
    x = arg[0]
    y = arg[1]
    return ((y - x ** 2) ** 2) + ((1. - x) ** 2)

def rosenbrock_grad(arg):
    x = arg[0]
    y = arg[1]
    dx = 2 * (2 * (x ** 3) - 2 * x * y + x - 1)
    dy = 2 * (y - (x ** 2))
    return np.array([dx, dy])


linear_step_choosers = [linear_step_chooser(cur_fun) for cur_fun in [dichotomy_method, golden_ratio_method, fib]]
backtrackings = [backtracking]
for cur_step_chooser in linear_step_choosers + backtrackings:
    trace = gradient_descent(sum_squares, sum_squares_grad,
                             np.array([-10, 20]), cur_step_chooser,
                             'grad', 1e-4)
    print('answer =', trace[-1], 'steps =', len(trace))


draw_function('sum_squares_golden_gradient',
              sum_squares,
              sum_squares_grad,
              start=np.array([-10, 20]),
              method=linear_step_chooser(golden_ratio_method),
              show_trace_maker=lambda trace: [trace[i] for i in range(len(trace)) if i % 200 == 0 or i < 10],
              x_min=-15,
              x_max=15,
              x_step=0.01,
              y_min=-10,
              y_max=30,
              y_step=0.01,
              levels=1000
             )


draw_function('rosenbrok_golden_gradient',
              rosenbrock,
              rosenbrock_grad,
              start=np.array([-10, 20]),
              method=linear_step_chooser(golden_ratio_method),
              show_trace_maker=lambda trace: [trace[i] for i in range(len(trace)) if i % 200 == 0 or i < 10],
              x_min=-15,
              x_max=15,
              x_step=0.01,
              y_min=-10,
              y_max=30,
              y_step=0.01,
              levels=1000
             )


def make_plot(step_chooser, name):
    n_vars = [5, 10, 20, 50, 100]
    condition_numbers = np.linspace(1, 1000, 50)
    plt.figure()
    for var in tqdm(n_vars):
        iter_numbers = [number_of_iters(cond, var, step_chooser) for cond in condition_numbers]
        plt.plot(condition_numbers, iter_numbers, label='n={}'.format(var))

    plt.xlabel('Число обусловленности')
    plt.ylabel('Число итераций')
    plt.legend()
    save(name, 'png')
    plt.show()
    print('end')



make_plot(linear_step_chooser(golden_ratio_method), 'golden_stat')