from logistic_regression import read_dataset, process_with_solver

X, y = read_dataset('geyser.csv')
process_with_solver(X, y, 'gradient', 0.1, 0.01)
process_with_solver(X, y, 'newton', 0.1, 0.01)