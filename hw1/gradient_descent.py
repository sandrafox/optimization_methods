from math import sqrt

import matplotlib
from jedi.api.refactoring import inline


import numpy.linalg
from one_dimesional_search import dichotomy_method, golden_ratio_method, fib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import os


def gradient_descent(f, f_grad, start_arg, step_chooser, stop_criterion, eps=1e-5):
    assert stop_criterion in {'arg', 'value', 'grad'}
    cur_arg = start_arg
    cur_value = f(cur_arg)
    trace = [cur_arg]
    start_grad = numpy.linalg.norm(f_grad(start_arg)) ** 2
    while True:
        cur_grad = f_grad(cur_arg)
        cur_step = step_chooser(f, cur_grad, cur_arg)
        next_arg = cur_arg - cur_step * cur_grad
        next_value = f(next_arg)
        trace.append(next_arg)

        if (stop_criterion == 'arg' and numpy.linalg.norm(next_arg - cur_arg) < eps) or \
                (stop_criterion == 'value' and abs(next_value - cur_value) < eps) or \
                (stop_criterion == 'grad' and numpy.linalg.norm(cur_grad) ** 2 < eps):
            return trace
        cur_arg = next_arg
        cur_value = next_value


def backtracking(f, f_grad, left_border, delta = 1, eps=0.5, multiplier=2):
    def linear_optimization_problem(k):
        return f(left_border - k * f_grad)

    start_value = linear_optimization_problem(0.)
    start_grad = numpy.linalg.norm(f_grad) ** 2
    #_, cur_delta = line_search(linear_optimization_problem, 0.)
    cur_delta = delta
    while linear_optimization_problem(cur_delta) > start_value - eps * cur_delta * start_grad:
        cur_delta /= 2
    return cur_delta


def linear_step_chooser(method):
    def result(f, grad, arg):
        def linear_optimization_problem(k):
            return f(arg - k * grad)

        left_border = 0.
        left_border, right_border = line_search(linear_optimization_problem, left_border)
        answer, _ = method(linear_optimization_problem, left_border, right_border)
        return answer

    return result


def line_search(f, left_border, start_delta=0.01, eps=1e-3, multiplier=2):
    cur_value = f(left_border)
    cur_delta = start_delta
    right_border = left_border + cur_delta
    next_value = f(right_border)
    if next_value > cur_value:
        cur_delta *= -1
        right_border = left_border + cur_delta
        next_value = f(right_border)
    if next_value > cur_value:
        return right_border, left_border - cur_delta
    while next_value <= cur_value + eps:
        cur_value = next_value
        cur_delta *= multiplier
        right_border += cur_delta
        next_value = f(right_border)
    if left_border < right_border:
        return left_border, right_border
    else:
        return right_border, left_border


def rosenbrock_f(arg):
    x = arg[0]
    y = arg[1]
    return 100 * ((y - x ** 2) ** 2) + ((1. - x) ** 2)

def rosenbrock_grad(arg):
    x = arg[0]
    y = arg[1]
    dx = 2 * (200 * (x ** 3) - 200 * x * y + x - 1)
    dy = 200 * (y - (x ** 2))
    return numpy.array([dx, dy])





def save(name='', fmt='png'):
    pwd = os.getcwd()
    iPath = 'pictures/{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)


def draw_function(name, f, f_grad, start, method, show_trace_maker, x_min, x_max, x_step, y_min, y_max, y_step, levels):
    trace = gradient_descent(f, f_grad, start, method, stop_criterion='grad', eps=1e-10)
    print('Answer =', trace[-1])
    print(len(trace), 'steps')
    trace_to_show = show_trace_maker(trace)

    x_s = np.arange(x_min, x_max, x_step)
    y_s = np.arange(y_min, y_max, y_step)
    z_s = np.array([[f(np.array([x, y])) for x in x_s] for y in y_s])

    plt.figure()
    cs = plt.contour(x_s, y_s, z_s, levels=levels)
    # plt.clabel(cs)
    for i in tqdm(range(len(trace_to_show) - 1)):
        cur_point = trace_to_show[i]
        next_point = trace_to_show[i + 1]
        plt.scatter([cur_point[0]], [cur_point[1]])
        plt.plot([cur_point[0], next_point[0]], [cur_point[1], next_point[1]])
    plt.grid()
    save(name, fmt='png')
    plt.show()





def create_matrix(condition_number, n):
    r = sqrt(condition_number)
    A = np.random.randn(n, n)
    u, s, v = np.linalg.svd(A)
    h, l = np.max(s), np.min(s)  # highest and lowest eigenvalues (h / l = current cond number)

    # linear stretch: f(x) = a * x + b, f(h) = h, f(l) = h/r, cond number = h / (h/r) = r
    def f(x):
        return h * (1 - ((r - 1) / r) / (h - l) * (h - x))

    new_s = f(s)
    new_A = (u * new_s) @ v.T  # make inverse transformation (here cond number is sqrt(k))
    new_A = new_A @ new_A.T  # make matrix symmetric and positive semi-definite (cond number is just k)
    assert np.isclose(np.linalg.cond(new_A), condition_number)
    return new_A


def number_of_iters(cond, n_vars, step_chooser, n_checks=50):
    avg_iters = 0
    for _ in range(n_checks):
        A = create_matrix(cond, n_vars)
        b = np.random.randn(len(A))
        init_x = np.random.randn(len(A))
        f = lambda x: x.dot(A).dot(x) - b.dot(x)
        f_grad = lambda x: (A + A.T).dot(x) - b

        # print(f(np.linalg.inv(A+A.T).dot(b))) -- optimal value

        trace = gradient_descent(f, f_grad, init_x, step_chooser, 'value')

        avg_iters += len(trace)
    return avg_iters / n_checks


