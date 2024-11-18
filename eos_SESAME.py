import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.optimize import root_scalar, root, minimize
from tqdm import tqdm


def make_into_pair_array(arr1, arr2):
    # turns two multidimensional numpy arrays into a form that can be used with the scipy interpolator

    arr1, arr2 = np.nan_to_num(arr1), np.nan_to_num(arr2)

    if type(arr1) is np.ndarray and type(arr2) is np.ndarray:

        if arr1.ndim == 0:
            return np.array([arr1[()], arr2[0]])
        if arr2.ndim == 0:
            return np.array([arr1[0], arr2[()]])

        try:
            assert np.all(arr1.shape == arr2.shape)
        except AssertionError:
            print(f'arr1 = {arr1} \n arr1 shape = {arr1.shape}')
            print(f'arr2 = {arr2} \n arr2 shape = {arr2.shape}')
            assert np.all(arr1.shape == arr2.shape)

        assert arr1.ndim == 1 or arr1.ndim == 2

        arr = np.array([arr1, arr2])

        if arr1.ndim == 1:
            return np.transpose(arr, axes=(1, 0))
        elif arr1.ndim == 2:
            return np.transpose(arr, axes=(1, 2, 0))

    else:

        if type(arr1) is np.ndarray:
            if arr1.ndim == 1:
                arr1 = arr1[0]

        if type(arr2) is np.ndarray:
            if arr2.ndim == 1:
                arr2 = arr2[0]

        return np.array([arr1, arr2])


class eos:

    def __init__(self, path_to_SESAME_table, material_name, material_id):

        self.material_name = material_name
        self.material_id = material_id

        # reads eos data from file
        print(f'Loading EOS tables for {material_name}...')
        with open(path_to_SESAME_table) as f:

            # Skip the header
            for i in range(12):
                f.readline()

            # Skip the version date
            f.readline()

            self.num_rho, self.num_T = np.array(f.readline().split(), dtype=int)
            self.A2_u = np.empty((self.num_rho, self.num_T))
            self.A2_P = np.empty((self.num_rho, self.num_T))
            self.A2_c = np.empty((self.num_rho, self.num_T))
            self.A2_s = np.empty((self.num_rho, self.num_T))

            self.A1_rho = np.array(f.readline().split(), dtype=float)
            self.A1_T = np.array(f.readline().split(), dtype=float)

            for i_T in range(self.num_T):
                for i_rho in range(self.num_rho):
                    (
                        self.A2_u[i_rho, i_T],
                        self.A2_P[i_rho, i_T],
                        self.A2_c[i_rho, i_T],
                        self.A2_s[i_rho, i_T],
                    ) = np.array(f.readline().split(), dtype=float)

        print('Table loaded.')

        # find minimum and maximum values
        self.rho_min, self.rho_max = np.min(self.A1_rho), np.max(self.A1_rho)
        self.T_min, self.T_max = np.min(self.A1_T), np.max(self.A1_T)
        # setting a minimum pressure for the EOS at 1 Pa and a maximum at 1 TPa
        self.P_min, self.P_max = np.maximum(1, np.min(self.A2_P)), np.minimum(np.max(self.A2_P), 1e12)
        # setting a minimum entropy for the EOS at 1 J/K/kg
        self.s_min, self.s_max = np.maximum(np.min(self.A2_s), 1), np.minimum(np.max(self.A2_s), 10000)

        # create log arrays for rho and P

        self.A1_log_rho = np.log10(self.A1_rho)
        self.A2_log_P = np.log10(self.A2_P)

        # create interpolators for P and s

        print('Initializing interpolators...')

        interpolator_method = 'linear'

        self.P_interpolator = RegularGridInterpolator(
            (self.A1_rho, self.A1_T),
            self.A2_P,
            method=interpolator_method,
            fill_value=None,
            bounds_error=True
        )

        self.s_interpolator = RegularGridInterpolator(
            (self.A1_rho, self.A1_T),
            self.A2_s,
            method=interpolator_method,
            fill_value=None,
            bounds_error=True
        )

        # invert the P and s tables to get interpolators for rho(P, T), rho(P, s), T(P, s)

        def find_x_for_yz(y, z, interp_z, x_min, x_max):
            # function to invert z(x, y) to x(y, z)
            try:
                sol = root_scalar(lambda x: interp_z((x, y)) - z, bracket=[x_min, x_max], method="brentq")
            except ValueError:
                z_guess_min, z_guess_max = interp_z((x_min, y)), interp_z((x_min, y))
                if np.abs(z_guess_min - z) / z < 0.01:
                    return x_min
                elif np.abs(z_guess_max - z) / z < 0.01:
                    return x_max
                else:
                    sol = root_scalar(lambda x: interp_z((x, y)) - z, x0=(x_min + x_max) / 2)
            return sol.root if sol.converged else np.nan

        def find_xy_for_zw(z, w, interp_z, interp_w, x0, y0):
            # function to invert z(x, y) and w(x, y) to x(z, w) and y(z, w)
            #error = lambda x: ((interp_z(x)[0] - z) / z) ** 2 + ((interp_w(x)[0] - w) / w) ** 2
            sol = root(lambda x: [interp_z(x)[0] - z, interp_w(x)[0] - w], x0=np.array([x0, y0]))
            #sol = minimize(error, x0=np.array([x0, y0]), bounds=((self.rho_min, self.rho_max), (self.T_min, self.T_max)))
            return sol.x if sol.success else np.array([np.nan, np.nan])

        n = 100

        self.A1_P = np.logspace(np.log10(self.P_min), np.log10(self.P_max), num=n)
        self.A1_s = np.logspace(np.log10(self.s_min), np.log10(self.s_max), num=n)

        self.A2_rho_PT = np.empty((n, self.num_T), dtype=float)
        # self.A2_rho_Ps = np.empty((n, n), dtype=float)
        # self.A2_T_Ps = np.empty((n, n), dtype=float)

        A1_rho_Ps, A1_T_Ps = [], []
        Ps_interpolation_points = []

        for i, P in tqdm(enumerate(self.A1_P)):
            for j, s in enumerate(self.A1_s):
                combined_error = np.sqrt(((self.A2_P - P) / P) ** 2 + ((self.A2_s - s) / s) ** 2)
                log_min_error = np.nanmin(np.log10(combined_error))

                if log_min_error < -2:
                    multi_index = np.unravel_index(np.argmin(combined_error), combined_error.shape)
                    rho_guess, T_guess = self.A1_rho[multi_index[0]], self.A1_T[multi_index[1]]

                    try:

                        rho, T = find_xy_for_zw(P, s, self.P_interpolator, self.s_interpolator, rho_guess, T_guess)
                        P_test, s_test = self.P_rhoT(rho, T), self.s_rhoT(rho, T)
                        P_error, s_error = np.abs(P - P_test) / P, np.abs(s - s_test) / s

                        assert P_error < 0.05 and s_error < 0.05

                        Ps_interpolation_points.append([P, s])
                        A1_rho_Ps.append(rho)
                        A1_T_Ps.append(T)

                    except OutOfEOSRangeException:
                        pass
                    except ValueError:
                        pass

        Ps_interpolation_points = np.array(Ps_interpolation_points)
        A1_rho_Ps, A1_T_Ps = np.array(A1_rho_Ps), np.array(A1_T_Ps)

        for i, P in tqdm(enumerate(self.A1_P)):
            for j, T in enumerate(self.A1_T):
                rho = find_x_for_yz(T, P, self.P_interpolator, self.rho_min, self.rho_max)
                P_test = self.P_rhoT(rho, T)
                P_error = np.abs(P - P_test) / P

                assert P_error < 0.05

                self.A2_rho_PT[i, j] = rho

        self.rho_PT_interpolator = RegularGridInterpolator(
            (self.A1_P, self.A1_T),
            self.A2_rho_PT,
            method=interpolator_method,
            fill_value=np.nan
        )

        self.rho_Ps_interpolator = LinearNDInterpolator(
            Ps_interpolation_points,
            A1_rho_Ps,
            fill_value=np.nan
        )

        self.T_Ps_interpolator = LinearNDInterpolator(
            Ps_interpolation_points,
            A1_T_Ps,
            fill_value=np.nan
        )

        print('Interpolators initialized.')

    def input_check(self, rho, T, P, s):

        if rho is not None:
            if np.any(rho > self.rho_max) or np.any(rho < self.rho_min):
                raise OutOfEOSRangeException('rho')
        if P is not None:
            if np.any(P > self.P_max) or np.any(P < self.P_min):
                raise OutOfEOSRangeException('P')
        if T is not None:
            if np.any(T > self.T_max) or np.any(T < self.T_min):
                raise OutOfEOSRangeException('T')
        if s is not None:
            if np.any(s > self.s_max) or np.any(s < self.s_min):
                raise OutOfEOSRangeException('s')

    def P_rhoT(self, rho, T):
        self.input_check(rho, T, None, None)
        return self.P_interpolator(make_into_pair_array(rho, T))

    def s_rhoT(self, rho, T):
        self.input_check(rho, T, None, None)
        return self.s_interpolator(make_into_pair_array(rho, T))

    def s_PT(self, P, T):
        self.input_check(None, T, P, None)
        return self.s_rhoT(self.rho_PT(P, T), T)

    def rho_PT(self, P, T):
        self.input_check(None, T, P, None)
        return self.rho_PT_interpolator(make_into_pair_array(P, T))

    def rho_Ps(self, P, s):
        self.input_check(None, None, P, s)
        return self.rho_Ps_interpolator(P, s)

    def T_Ps(self, P, s):
        self.input_check(None, None, P, s)
        return self.T_Ps_interpolator(P, s)


class OutOfEOSRangeException(Exception):

    def __int__(self, variable):
        message = f'{variable} value out of range'
        super.__init__(message)


#WATER = eos('data/AQUA_H20.txt', 'Water', 304)
FORSTERITE = eos('data/ANEOS_forsterite_S19.txt', 'Forsterite', 400)
IRON = eos('data/ANEOS_iron_S20.txt', 'Iron', 401)
