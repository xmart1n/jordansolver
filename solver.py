import numpy as np
import sympy as sp

from solver_auxilary import *
import json

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


class Solver:
    def __init__(self):
        self.mx_datapack = None
        self.matrix = None

        self.clear_datapack()

    def clear_datapack(self):
        self.mx_datapack = {
            "eigenvalues_count": 0,
            "eigenvalues": [],
            "jordan_matrix": []
        }

    def processing_of_eigenvalues(self, e):
        proper_value, algebraic_multiplicity, eigenvectors = e

        b = np.array(self.matrix - proper_value * np.eye(self.matrix.shape[0]), dtype=np.int32)
        label_vectors, vectors_data = searching_Jordan_cell_vectors(b, algebraic_multiplicity)
        ladder_data = generate_ladder(b, label_vectors)
        get_transition_matrix = generate_transactions(ladder_data)

        return {
            "proper_value": proper_value.p,
            "algebraic_multiplicity": algebraic_multiplicity,  # размер J(e.v.)
            "eigenvectors": [tuple([vv.p for vv in v]) for v in eigenvectors],
            "B": vectors_data,
            "ladder": list(reversed(ladder_data)),
            "transition_matrix": get_transition_matrix
        }

    def process(self, mx):
        self.matrix = np.array(mx, dtype=np.int32)
        self.clear_datapack()

        mx_sp = sp.Matrix(self.matrix)
        handmade_transition_matrix = []

        eigvecs = mx_sp.eigenvects()
        for e in eigvecs:
            ev_data = self.processing_of_eigenvalues(e)
            self.mx_datapack["eigenvalues"].append(ev_data)
            handmade_transition_matrix += ev_data["transition_matrix"]

        P, J = mx_sp.jordan_form()
        htm = np.array(handmade_transition_matrix, dtype=np.int32).T.tolist()
        self.mx_datapack["eigenvalues_count"] = len(eigvecs)
        self.mx_datapack["jordan_matrix"] = to_list(J)
        self.mx_datapack["transition_matrix"] = to_list(P)
        self.mx_datapack["handmade_transition_matrix"] = htm
        self.mx_datapack["http"] = int(check(self.matrix, handmade_transition_matrix, to_list(
            J)))  # флаг, что сошлись библиотечный и рукописный переходы

    def get_datapack(self) -> str:
        return json.dumps(self.mx_datapack)


if __name__ == "__main__":
    s = Solver()
    s.process(MX)
    print(s.get_datapack())
