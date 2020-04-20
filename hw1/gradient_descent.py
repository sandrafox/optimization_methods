import numpy.linalg

def gradient_descent(f, f_grad, start_arg, step_chooser, stop_criterion, eps=1e-5):
    assert stop_criterion in {'arg', 'value', 'grad'}
    cur_arg = start_arg
    cur_value = f(cur_arg)
    trace = [cur_arg]
    start_grad = numpy.linalg.norm(f_grad(start_arg)) ** 2
    while True:
        cur_grad = f_grad(cur_arg)
        cur_step = backtracking(f, f_grad, 0)
        next_arg = cur_arg - cur_step * cur_grad
        next_value = f(next_arg)
        trace.append(next_arg)

        if (stop_criterion == 'arg' and numpy.linalg.norm(next_arg - cur_arg) < eps) or \
                (stop_criterion == 'value' and abs(next_value - cur_value) < eps) or \
                (stop_criterion == 'grad' and numpy.linalg.norm(cur_grad) ** 2 < eps * start_grad):
            return trace
        cur_arg = next_arg
        cur_value = next_value


def backtracking(f, f_grad, left_border, start_delta=0.01, eps=1e-5, multiplier=2):
    start_value = f(left_border)
    start_grad = numpy.linalg.norm(f_grad(left_border)) ** 2
    right_border = left_border + start_delta
    cur_delta = start_delta
    while f(right_border) < start_value + eps * cur_delta * start_grad:
        cur_delta *= multiplier
        right_border += cur_delta
    return cur_delta
