import numpy as np
import sympy as sp

import pprint


"""
MX = [
    [0, 1, 0, 0],
    [-1, 2, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 2],
]


"""
MX = [
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [-1, -1, 0, 0, -1],
    [1, 1, 0, 0, 1],
    [0, -1, 0, 0, 0],
]


def to_list(mx) -> list:
    # Преобразует в список
    sz = mx.shape
    return np.array(mx).reshape(sz[1], sz[0]).tolist()


def get_own_variables(mx: np.ndarray):
    mx_sp = sp.Matrix(mx)
    eigvecs = mx_sp.eigenvects()
    return eigvecs


def searching_Jordan_cell_vectors(mx: np.ndarray, target):
    b_pow = 1
    vectors_count = 0
    proper_values = [[]]

    printable = []

    while 1:
        ev = get_own_variables(mx)[0]
        mx_rang = np.linalg.matrix_rank(mx)
        pv = [tuple(v) for v in ev[2]]
        n_info = {"B": b_pow, "matrix": mx, "vectors": []}
        printable.append(n_info)

        for i in range(len(pv)):
            vectors_set = sum(proper_values, []) + [pv[i]]
            matrix = np.array(vectors_set, dtype=np.int32)

            if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
                proper_values[-1] += [pv[i]]
                printable[-1]["vectors"] += [pv[i]]
                vectors_count += 1
                if target == vectors_count:
                    break

        b_pow += 1
        next_mx = np.linalg.matrix_power(mx, b_pow)
        if mx_rang == np.linalg.matrix_rank(next_mx):
            break
        mx = next_mx
        proper_values += [[]]

    if not proper_values[-1]:
        proper_values = proper_values[:-1]

    return proper_values, printable


def generate_ladder(b, label_vectors):
    l = [[[] for _ in range(len(i))] for i in label_vectors]
    high = len(label_vectors)

    for i in range(len(label_vectors[-1])):
        l[-1][i] = label_vectors[-1][i]
        for j in range(high - 2, -1, -1):
            vector = np.array(l[j + 1][i], dtype=np.int32).reshape(-1, 1)
            l[j][i] = tuple((b @ vector).reshape(-1))

    for i in range(high - 2, -1, -1):
        for j in range(len(label_vectors[i])):
            if not l[i][j]:
                row = [v for v in label_vectors[i] if v not in l[i]]

                if row:
                    l[i][j] = row[0]
                    for k in range(i - 1, -1, -1):
                        vector = np.array(l[k + 1][i], dtype=np.int32).reshape(-1, 1)
                        l[k][i] = tuple((b @ vector).reshape(-1))

    return l


def generate_transactions(ladder):
    vectors = []
    size0 = (len(ladder), len(ladder[0]))
    vsize = max(size0) + 1
    size = (vsize, vsize, len(ladder[0][0]))
    arr = np.zeros((size), dtype=np.int32)
    for i in range(len(ladder)):
        for j in range(len(ladder[i])):
            arr[i][j] = np.array(ladder[i][j], dtype=np.int32)
    arr = np.transpose(arr, (1, 0, 2))

    x = 0
    y = 0
    while np.any(arr[x][y] != 0) and x < vsize:
        while np.any(arr[x][y] != 0) and y < vsize:
            vectors.append(tuple(arr[x][y].tolist()))
            y += 1
        x += 1
        y = 0
    return vectors


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
        label_vectors, vectors_data = searching_Jordan_cell_vectors(b, algebraic_multiplicity)
        ladder_data = generate_ladder(b, label_vectors)
        get_transition_matrix = generate_transactions(ladder_data)

        return {
            "proper_value": proper_value,
            "algebraic_multiplicity": algebraic_multiplicity,  # размер J(e.v.)
            "eigenvectors": [tuple(v) for v in eigenvectors],
            "B": vectors_data,
            "ladder": list(reversed(ladder_data)),
            "transition_matrix": get_transition_matrix
        }

    def main(self):
        mx_sp = sp.Matrix(self.matrix)
        handmade_transition_matrix = []

        eigvecs = mx_sp.eigenvects()
        for e in eigvecs:
            ev_data = self.processing_of_eigenvalues(e)
            self.mx_datapack["eigenvalues"].append(ev_data)
            handmade_transition_matrix += ev_data["transition_matrix"]

        P, J = mx_sp.jordan_form()
        self.mx_datapack["eigenvalues_count"] = len(eigvecs)
        self.mx_datapack["jordan_matrix"] = to_list(J)
        self.mx_datapack["transition_matrix"] = to_list(P)
        self.mx_datapack["handmade_transition_matrix"] = handmade_transition_matrix

        return self.mx_datapack


if __name__ == "__main__":
    s = Solver(MX)

    dp = s.main()
    pprint.pprint(dp)
