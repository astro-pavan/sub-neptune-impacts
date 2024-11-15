import matplotlib.pyplot as plt
import numpy as np
from numerical_methods import solve_ivp_rk4, root_finder
from scipy.optimize import root_scalar, root
import woma
import pandas as pd
import woma.eos.sesame as eos

from adiabat import T_profile, s_P_T

material_names = {
    304: 'Water',
    400: 'Forsterite',
    401: 'Iron'
}

G = 6.67e-11
R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg

woma.load_eos_tables(['ANEOS_forsterite', 'ANEOS_iron', 'AQUA'])

# rho = eos.A1_rho_AQUA
# T = eos.A1_T_AQUA
# logP = eos.A2_log_P_AQUA
#
# X, Y = np.meshgrid(T, rho)
#
# plt.contourf(X, Y, logP, 200, cmap='turbo')
# plt.yscale('log')
# plt.xlim([0, 2000])
# plt.colorbar()
# plt.show()


def integrate_material(m_0, m_1, r_0, P_0, T_0, mat_id):

    print(f'Integrating {material_names[mat_id]}...')

    # step 1 - make adiabatic thermal profile

    s_0 = s_P_T(P_0, T_0, mat_id)
    T = T_profile(np.logspace(np.log10(P_0), 14, num=400), s_0, mat_id)

    # step 2 - set up EOS

    rho = lambda P: woma.rho_P_T(P, T(P)[()], mat_id)
    A1_rho = lambda P: woma.A1_rho_P_T(P, T(P), np.full_like(P, mat_id))

    # step 3 - set up structure equations for solver

    def dr_dm(m, r, P):
        assert P > 0
        return 1 / (4 * np.pi * (r ** 2) * rho(P))

    def dP_dm(m, r, P):
        return (- G * m) / (4 * np.pi * (r ** 4))

    def f(t, y):
        return np.array([dr_dm(t, y[0], y[1]), dP_dm(t, y[0], y[1])])

    # step 4 - solve structure equations

    h = (m_1 - m_0) / 2000

    terminate = lambda t, y: y[1] < 0 or y[0] < 0 or T(y[1]) < 0 or rho(y[1]) < 0

    t, y = solve_ivp_rk4(f, (m_0, m_1), [r_0, P_0], h, terminate_condition=terminate)

    m = np.array(t)
    r_solution = np.array(y[:, 0])
    P_solution = np.array(y[:, 1])
    T_solution = T(P_solution)
    rho_solution = A1_rho(P_solution)

    return m, r_solution, P_solution, T_solution, rho_solution


def integrate_planet(M_planet, R_planet, P_surface, T_surface, materials, material_fractions, return_df=False):

    print('Integrating planet...')

    m_start, m_end = M_planet, 0
    r_start = R_planet
    P_start, T_start = P_surface, T_surface
    r_end, P_end, T_end = 0, 0, 0

    m_solution, r_solution, P_solution, T_solution, rho_solution = [], [], [], [], []

    integration_success = False

    for i in range(len(material_fractions)):

        f = material_fractions[i]
        mat_id = materials[i]
        m_end = m_start - (f * M_planet)

        m, r, P, T, rho = integrate_material(m_start, m_end, r_start, P_start, T_start, mat_id)

        m_end, r_end, P_end, T_end = m[-1], r[-1], P[-1], T[-1]

        m_solution.append(m[:-1])
        r_solution.append(r[:-1])
        P_solution.append(P[:-1])
        T_solution.append(T[:-1])
        rho_solution.append(rho[:-1])

        m_start, r_start, P_start, T_start = m_end, r_end, P_end, T_end

        if r_end < 0 or P_end < 0:
            break

    try:
        m_solution = np.hstack(m_solution)
        r_solution = np.hstack(r_solution)
        P_solution = np.hstack(P_solution)
        T_solution = np.hstack(T_solution)
        rho_solution = np.hstack(rho_solution)
    except ValueError:
        pass

    df = pd.DataFrame({'m': m_solution, 'r': r_solution,
                       'P': P_solution, 'T': T_solution, 'rho': rho_solution})

    if 100 > r_end > 0:
        integration_success = True
    else:
        print(f'Integration error: r_end = {r_end:.2e} m, P_end = {P_end:.2e} Pa')

    print(f'Planet integration {"successful!" if integration_success else "failed."}')

    if return_df:
        return integration_success, r_end, df
    else:
        return integration_success, r_end


def create_2_layer_planet(M_planet, R_planet, P_surface, T_surface, materials):

    fun = lambda f_core: integrate_planet(M_planet, R_planet, P_surface, T_surface, materials, [1 - f_core, f_core])

    mid, _ = root_finder(fun, 0.01, 0.5)

    df = integrate_planet(6e24, 6.4e6, 1e5, 1500, [400, 401], [1 - mid, mid], return_df=True)[2]

    plt.plot(df['r'] / R_earth, df['m'] / M_earth)
    plt.show()


def create_2_layer_planet_v2(M_planet, R_planet, P_surface, T_surface, materials):
    fun = lambda f_core: integrate_planet(M_planet, R_planet, P_surface, T_surface, materials, [1 - f_core, f_core])[1]

    res = root_scalar(fun, bracket=(0.01, 0.99), method='bisect')

    f = res.root()

    df = integrate_planet(6e24, 6.4e6, 1e5, 1500, [400, 401], [1 - f, f], return_df=True)[2]

    plt.plot(df['r'] / R_earth, df['m'] / M_earth)
    plt.show()


def create_3_layer_planet(M_planet, R_planet, P_surface, T_surface, materials):

    fun = lambda x:\
        integrate_planet(M_planet, R_planet, P_surface, T_surface, materials, [1 - x[0] - x[1], x[0], x[1]])

    res = root(fun, [0.1, 0.1], method='df-sane')

    print(res.message)

    f1 = res.x[0]
    f2 = res.x[1]

    print(f'Core: {f2:.2%} Mantle: {f1:.2%} Ocean: {1 - f1 - f2:.2%}')

    df = integrate_planet(M_planet, R_planet, P_surface, T_surface, materials, [1 - f1 - f2, f1, f2], return_df=True)[2]

    plt.plot(df['r'] / R_earth, df['m'] / M_earth)
    plt.show()


# create_2_layer_planet_v2(6e24, 6.4e6, 1e5, 1500, [400, 401])

create_3_layer_planet(4.8 * M_earth, 2.1 * R_earth, 1e5, 400, [304, 400, 401])

# m, r, P, T, rho = integrate_material(5 * M_earth, (5 * 0.7) * M_earth, 2 * R_earth, 1e5, 400, 304)
#
# plt.plot(r / R_earth, T)
# plt.show()

# df = integrate_planet(5 * M_earth, 2 * R_earth, 1e5, 1500, [304, 400, 401], [0.1, 0.6, 0.3], return_df=True)[2]
#
# plt.plot(df['r'] / R_earth, df['rho'] / 1000)
# plt.yscale('log')
# plt.show()
