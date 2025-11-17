import numpy as np
import sympy as sp

import pprint

MX = [
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [-1, -1, 0, 0, -1],
    [1, 1, 0, 0, 1],
    [0, -1, 0, 0, 0],
]


def calculate_gk(mx) -> int:
    # Находит геометрическую кратность матрицы
    return mx.shape[0] - mx.rank()


def to_list(mx) -> list:
    # Преобразует в список
    sz = mx.shape
    return np.array(mx).reshape(sz[1], sz[0]).tolist()


def get_own_variables(mx: np.ndarray):
    mx_sp = sp.Matrix(mx)
    eigvecs = mx_sp.eigenvects()
    return eigvecs


def check_linear_independence(vectors):
    return


def searching_for_Jordan_cell_vectors(mx: np.ndarray):
    pow = 1
    proper_values = []
    leap = -1

    while 1:
        ev = get_own_variables(mx)[0]  # fixme ВООБЩЕ НЕ УВЕРЕН
        mx_rang = np.linalg.matrix_rank(mx)
        pv = [tuple(v) for v in ev[2]]

        proper_values_bi = []
        sweep = sum(proper_values, [])
        for i in range(min(mx_rang, len(pv))):
            matrix = np.array(sweep + proper_values_bi + [pv[i]], dtype=np.int32)
            if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
                proper_values_bi += [pv[i]]

        if proper_values_bi:
            proper_values += [proper_values_bi]

        pow += 1
        next_mx = np.linalg.matrix_power(mx, pow)

        if mx_rang == np.linalg.matrix_rank(next_mx):
            break
        mx = next_mx

    pass


class Solver:
    def __init__(self, matrix):
        self.mx_datapack = {
            "eigenvalues_count": 0,
            "eigenvalues": [],
            "jordan_matrix": []
        }

        self.matrix = np.array(matrix, dtype=np.int32)

    def processing_of_eigenvalues(self, e):
        proper_value, algebraic_multiplicity, eigenvectors = e

        b = np.array(self.matrix - proper_value * np.eye(self.matrix.shape[0]), dtype=np.int32)
        searching_for_Jordan_cell_vectors(b)

        return {
            "proper_value": proper_value,
            "algebraic_multiplicity": algebraic_multiplicity,  # размер J(e.v.)
            "eigenvectors": [tuple(v) for v in eigenvectors],
            "B": to_list(b)
        }

    def main(self):
        mx_sp = sp.Matrix(self.matrix)

        eigvecs = mx_sp.eigenvects()
        for e in eigvecs:
            ev_data = self.processing_of_eigenvalues(e)
            self.mx_datapack["eigenvalues"].append(ev_data)

        P, J = mx_sp.jordan_form()
        self.mx_datapack["eigenvalues_count"] = len(eigvecs)
        self.mx_datapack["jordan_matrix"] = to_list(J)
        self.mx_datapack["transition_matrix"] = to_list(P)

        return self.mx_datapack


if __name__ == "__main__":
    s = Solver(MX)

    dp = s.main()
    pprint.pprint(dp)
