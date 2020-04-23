import matplotlib.pyplot as plt
from scipy import integrate
from tqdm.notebook import tqdm
from one_dimesional_search import dichotomy_method, golden_ratio_method, fib

from gradient_descent import save


def make_plots(name, method, f, left_border, right_border, params, retry_count=100, params_is_eps=True):
    answers = []
    second_params = []
    for param in params:
        mean_answer = 0.
        mean_second_param = 0.

        for _ in range(retry_count):
            answer, second_param = method(f, left_border, right_border, param)
            mean_answer += answer
            mean_second_param += second_param
        answers.append(mean_answer / retry_count)
        second_params.append(mean_second_param / retry_count)

    plt.figure()
    plt.title('Number of iterations and eps')
    if params_is_eps:
        plt.xlabel('lg(eps)')
        plt.ylabel('Number of iterations')
        plt.semilogx(params, second_params)
    else:
        plt.xlabel('Number of iterations')
        plt.ylabel('lg(eps)')
        plt.semilogy(params, second_params)
    plt.grid()
    save(name, 'png')
    plt.show()


def f(x): # f(x) = x^2
    integral, err = integrate.quad(lambda t: 2 * t, 0, x)
    return integral


e_s = [10 ** x for x in range(-10, 0)]
make_plots('one_dimensional_dichotomy',dichotomy_method, f, -1400, 8800, e_s, params_is_eps=True)
make_plots('one_dimensional_golden',golden_ratio_method, f, -1400, 8800, e_s, params_is_eps=True)
ns = [10 * i for i in range(1, 11)]
make_plots('one_dimensional_fib', fib, f, -1400, 8800, ns, params_is_eps=False)