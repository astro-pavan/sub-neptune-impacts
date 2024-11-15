import numpy as np


def rk4_step(f, t, y, h):
    """
    Perform a single RK4 step.

    Parameters:
    - f: Function that returns dy/dt, with signature f(t, y)
    - t: Current time
    - y: Current value of the dependent variable (array-like)
    - h: Step size

    Returns:
    - y_next: Approximated value of y at time t + h
    """
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    y_next = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next


def solve_ivp_rk4(f, t_span, y0, h, terminate_condition=lambda t, y: True):
    """
    Solve an initial value problem (IVP) using the RK4 method.

    Parameters:
    - f: Function that returns dy/dt, with signature f(t, y)
    - t_span: Tuple (t0, tf) specifying the start and end times
    - y0: Initial value of the dependent variable (array-like)
    - h: Step size (positive; the function adjusts for direction)

    Returns:
    - t_values: Array of time values
    - y_values: Array of y values at each time step
    """
    t0, tf = t_span

    # Adjust step size for direction of integration
    if t0 > tf:
        h = -abs(h)  # Reverse direction if integrating backward
    else:
        h = abs(h)  # Ensure positive step size for forward integration

    t_values = [t0]
    y_values = [y0]

    t = t0
    y = np.array(y0, dtype=float)

    while (t < tf and h > 0) or (t > tf and h < 0):
        # Adjust step size if we're about to overshoot tf
        if (h > 0 and t + h > tf) or (h < 0 and t + h < tf):
            h = tf - t

        try:
            y = rk4_step(f, t, y, h)
        except AssertionError:
            print('Assertion error.')
            break

        t += h

        t_values.append(t)
        y_values.append(y.copy())

        if terminate_condition(t, y):
            break

    return np.array(t_values), np.array(y_values)


def root_finder(func, x0, x1, tol=1e-6, max_iter=100):
    """
    Finds the root of a function that returns (boolean, float) using the bisection method.

    Parameters:
    - func: A function that takes a float x and returns (bool, float).
            The bool indicates if the root is found.
    - x0: The lower bound of the search interval.
    - x1: The upper bound of the search interval.
    - tol: Convergence tolerance for the root-finding.
    - max_iter: Maximum number of iterations allowed.

    Returns:
    - root: The approximated root value.
    - f_value: The value of the function at the root.
    """
    for i in range(max_iter):
        mid = (x0 + x1) / 2
        is_converged, f_mid = func(mid)

        # Check if the root is found
        if is_converged or abs(f_mid) < tol:
            return mid, f_mid

        # Update the search interval
        _, f_x0 = func(x0)
        if f_mid * f_x0 < 0:
            x1 = mid
        else:
            x0 = mid

    raise ValueError(f"Root finding did not converge within {max_iter} iterations.")
