import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline
import woma

woma.load_eos_tables(['ANEOS_forsterite', 'ANEOS_iron', 'AQUA'])

s_P_T = lambda P, T, mat_id: woma.s_rho_T(woma.rho_P_T(P, T, mat_id), T, mat_id)
A1_s_P_T = lambda P, T, mat_id: woma.A1_s_rho_T(woma.A1_rho_P_T(P, T, np.full_like(P, mat_id)), T, np.full_like(P, mat_id))


# Function to find T(P) for a given s
def T_profile(P_values, s_value, mat_id):
    T_values = []

    for P in P_values:
        # Define the function whose root we're searching for at each P
        def equation(T):
            return s_P_T(P, T, mat_id) - s_value

        try:
        # Use root-finding to solve s(P, T) = s_value
            sol = root_scalar(equation, bracket=[1, 1e5], method='bisect')  # Adjust bracket based on expected T range
        except ValueError:
            last_T = T_values[-1]
            sol = root_scalar(equation, bracket=[last_T / 2, last_T * 2], method='bisect')

        assert np.abs(s_P_T(P, sol.root, mat_id) - s_value) / s_value < 0.01

        if sol.converged:
            T_values.append(sol.root)
        else:
            T_values.append(np.nan)

    profile = CubicSpline(P_values, T_values)

    return profile
