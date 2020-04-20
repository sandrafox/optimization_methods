import math


def dichotomy_method(f, left_border, right_border, eps=1e-5):
    iterations = 0
    delta = eps / 4
    while right_border - left_border > eps:
        iterations += 1
        middle = (left_border + right_border) / 2
        x1 = middle - delta
        x2 = middle + delta

        f1 = f(x1)
        f2 = f(x2)

        if f1 < f2:
            right_border = x2
        elif f1 > f2:
            left_border = x1
        else:
            return middle, iterations
    return (left_border + right_border) / 2, iterations


def golden_ratio_method(f, left_border, right_border, eps=1e-5):
    phi = (1 + math.sqrt(5)) / 2
    iterations = 0

    interval_len = right_border - left_border
    x1 = left_border + (2 - phi) * interval_len
    x2 = right_border - (2 - phi) * interval_len
    f1 = f(x1) #left
    f2 = f(x2) #right

    while interval_len > eps:
        iterations += 1
        if f1 < f2:
            right_border = x2
            x2 = x1
            f2 = f1
            interval_len = right_border - left_border
            x1 = left_border + (2 - phi) * interval_len
            f1 = f(x1)
        else:
            left_border = x1
            x1 = x2
            f1 = f2
            interval_len = right_border - left_border
            x2 = right_border - (2 - phi) * interval_len
            f2 = f(x2)
    return (right_border + left_border) / 2, iterations


def fib(f, left_border, right_border, n=60):
    fibs = [1, 1]
    while len(fibs) < n + 1:
        fibs.append(fibs[-1] + fibs[-2])

    interval_len = right_border - left_border
    x1 = left_border + (fibs[n - 2] / fibs[n]) * interval_len #left
    x2 = left_border + (fibs[n - 1] / fibs[n]) * interval_len #right
    f1 = f(x1)
    f2 = f(x2)

    while n > 2:
        n -= 1
        if f1 < f2:
            right_border = x2
            x2 = x1
            f2 = f1
            interval_len = right_border - left_border
            x1 = left_border + (fibs[n - 2] / fibs[n]) * interval_len
            f1 = f(x1)
        else:
            left_border = x1
            x1 = x2
            f1 = f2
            interval_len = right_border - left_border
            x2 = left_border + (fibs[n - 1] / fibs[n]) * interval_len
            f2 = f(x2)
    return (right_border + left_border) / 2, right_border - left_border


def line_search(f, left_border, start_delta=0.01, eps=1e-3, multiplier=2):
    start_value = f(left_border)
    right_border = left_border + start_delta
    cur_delta = start_delta
    while f(right_border) <= start_value + eps:
        cur_delta *= multiplier
        right_border += cur_delta
    return right_border