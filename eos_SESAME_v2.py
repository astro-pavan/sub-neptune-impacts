import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, griddata
from scipy.optimize import root_scalar, root, minimize, basinhopping, dual_annealing
from tqdm import tqdm
from shapely.geometry import LineString


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


def find_curve_intersections(curve1, curve2):
    # Convert curves to Shapely LineString objects
    line1 = LineString(curve1)
    line2 = LineString(curve2)

    # Find intersections
    intersection = line1.intersection(line2)

    # Parse intersection results
    if intersection.is_empty:
        return []  # No intersections

    if intersection.geom_type == 'Point':
        return [(intersection.x, intersection.y)]  # Single intersection

    if intersection.geom_type == 'MultiPoint':
        x = intersection.bounds[0]
        y = intersection.bounds[1]
        return [(x, y)]

    return []  # Default case (e.g., no valid intersections)


class eos:

    def __init__(self, path_to_SESAME_table, material_name, material_id, load_tables=True):

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

        self.logP_interpolator = RegularGridInterpolator(
            (self.A1_log_rho, self.A1_T), self.A2_log_P,
            method=interpolator_method, fill_value=None, bounds_error=False
        )

        self.s_interpolator = RegularGridInterpolator(
            (self.A1_log_rho, self.A1_T), self.A2_s,
            method=interpolator_method, fill_value=None, bounds_error=False
        )

        X, Y = np.meshgrid(self.A1_T, self.A1_rho)

        if load_tables:
            rho_logPT_interpolation_points = np.load(f'data/extra/{material_name}-rho-logPT.npy')
            rhoT_logPs_interpolation_points = np.load(f'data/extra/{material_name}-rhoT-logPs.npy')

        else:
            # invert the P and s tables to get interpolators for rho(P, T), rho(P, s), T(P, s)

            n = 200
            A1_log_P = np.linspace(1, 12, num=n)
            A1_s = np.logspace(2, 5, num=n)

            rhoT_logPs_interpolation_points = []
            rho_logPT_interpolation_points = []

            log_P_contours = plt.contour(X, Y, np.log10(self.A2_P), A1_log_P)
            s_contours = plt.contour(X, Y, self.A2_s, A1_s)

            log_P_contour_points = []
            for collection in log_P_contours.collections:
                contour_lines = []
                for path in collection.get_paths():
                    vertices = path.vertices
                    contour_lines.append(vertices)
                log_P_contour_points.append(contour_lines)

            s_contour_points = []
            for collection in s_contours.collections:
                contour_lines = []
                for path in collection.get_paths():
                    vertices = path.vertices
                    contour_lines.append(vertices)
                s_contour_points.append(contour_lines)

            plt.close()

            for i_log_P, log_P in enumerate(A1_log_P):

                rho, T = log_P_contour_points[i_log_P][0][:, 1], log_P_contour_points[i_log_P][0][:, 0]
                rho_logPT_interpolation_points.append(np.array([rho, np.full_like(rho, log_P), T]).T)

                for i_s, s in enumerate(A1_s):

                    intersection = find_curve_intersections(log_P_contour_points[i_log_P][0], s_contour_points[i_s][0])

                    if len(intersection) > 0:
                        T, rho = intersection[0]
                        P = 10 ** log_P

                        P_test, s_test = self.P_rhoT(rho, T)[0], self.s_rhoT(rho, T)[0]
                        P_error, s_error = np.abs(P_test - P) / P, np.abs(s_test - s) / s

                        if P_error < 0.1 and s_error < 0.1:
                            rhoT_logPs_interpolation_points.append([rho, T, log_P, s])

            rho_logPT_interpolation_points = np.vstack(rho_logPT_interpolation_points)
            rhoT_logPs_interpolation_points = np.array(rhoT_logPs_interpolation_points)

            np.save(f'data/extra/{material_name}-rho-logPT.npy', rho_logPT_interpolation_points)
            np.save(f'data/extra/{material_name}-rhoT-logPs.npy', rhoT_logPs_interpolation_points)

        self.rho_logPT_interpolator = LinearNDInterpolator(
            rho_logPT_interpolation_points[:, 1:],
            rho_logPT_interpolation_points[:, 0],
            fill_value=np.nan,
            rescale=True
        )

        self.rho_logPs_interpolator = LinearNDInterpolator(
            rhoT_logPs_interpolation_points[:, 2:],
            rhoT_logPs_interpolation_points[:, 0],
            fill_value=np.nan,
            rescale=True
        )

        self.T_logPs_interpolator = LinearNDInterpolator(
            rhoT_logPs_interpolation_points[:, 2:],
            rhoT_logPs_interpolation_points[:, 1],
            fill_value=np.nan,
            rescale=True
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
        return 10 ** self.logP_interpolator(make_into_pair_array(np.log10(rho), T))

    def s_rhoT(self, rho, T):
        self.input_check(rho, T, None, None)
        return self.s_interpolator(make_into_pair_array(np.log10(rho), T))

    def s_PT(self, P, T):
        self.input_check(None, T, P, None)
        return self.s_rhoT(self.rho_PT(P, T), T)

    def rho_PT(self, P, T):
        self.input_check(None, T, P, None)
        return self.rho_logPT_interpolator(np.log10(P), T)

    def rho_Ps(self, P, s):
        self.input_check(None, None, P, s)

        if type(P) is np.ndarray and type(s) is float:
            s = np.full_like(P, s)

        res = self.rho_logPs_interpolator(np.log10(P), s)
        return res

    def T_Ps(self, P, s):
        self.input_check(None, None, P, s)
        return self.T_logPs_interpolator(np.log10(P), s)


class OutOfEOSRangeException(Exception):

    def __int__(self, variable):
        message = f'{variable} value out of range'
        super.__init__(message)

    # def _generate_message(self):
    #     return ""
    #
    # def __str__(self):
    #     return self._generate_message()


if __name__ == '__main__':
    forsterite = eos('data/ANEOS_forsterite_S19.txt', 'forsterite', 400, load_tables=False)
    water = eos('data/AQUA_H20.txt', 'water', 304, load_tables=False)
    iron = eos('data/ANEOS_iron_S20.txt', 'iron', 401, load_tables=False)
