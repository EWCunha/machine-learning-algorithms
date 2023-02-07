import numpy as np


def univariate_linear_regression(x, y) -> tuple:

    """
    Calculates prediction values using univariate linear regression. Based on
    the common equations of the univariate linear regression parameters from statistics:
    a = (sum(x_i * y_i) - n * avg(x) * avg(y)) / (sum(x ** 2) - n * avg(x) ** 2)

    Args:
        x: array-like independent variable.
        y: array-like dependent variable.

    Returns:
        Array-like prediction and R-squared value.

    Raises:
        ValueError: If x and y do not have the same length.
    """

    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("x and y have not the same length!")

    avg_x = sum(x) / len(x)
    avg_y = sum(y) / len(y)
    n = len(x)

    sum_x_y = np.dot(x, y)
    sum_x_2 = np.dot(x, x)
    y_minus_avg_y = y - avg_y
    SQT = np.dot(y_minus_avg_y, y_minus_avg_y)

    a = (sum_x_y - n * avg_x * avg_y) / (sum_x_2 - n * avg_x ** 2)
    b = avg_y - a * avg_x

    y_pred = a * x + b
    diff_vec = y_pred - y
    SQRes = np.dot(diff_vec, diff_vec)

    r_squared = (SQT - SQRes) / SQT

    return y_pred, r_squared
