import sympy as sp

try:
    from .solver_auxilary import *
except:
    from solver_auxilary import *

import json

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
"""
"""
MX = [
    [0, 0, 1, -1, 0, 1],
    [1, 0, -3, 1, -1, -3],
    [0, 1, 3, 0, 1, 2],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, -3],
    [0, 0, 0, 0, 1, 3]
]
"""


class Solver:
    def __init__(self):
        self.mx_datapack = None
        self.matrix = None

        self.clear_datapack()

    def clear_datapack(self):
        self.mx_datapack = {"eigenvalues": []}

    def processing_of_eigenvalues(self, e):
        proper_value, algebraic_multiplicity, eigenvectors = e

        n = self.matrix.rows
        b = self.matrix - proper_value * sp.eye(n)
        label_vectors, vectors_info = searching_Jordan_cell_vectors(
            b, algebraic_multiplicity
        )
        ladder_data, ladder_info = generate_ladder(b, label_vectors)
        get_transition_matrix, cels_sizes = generate_transactions(ladder_data)
        jcels = generate_cels(proper_value, cels_sizes)

        return {
            "proper_value": jlatex(proper_value),
            "algebraic_multiplicity": jlatex(algebraic_multiplicity),  # размер J(e.v.)
            "eigenvectors": [jlatex(v) for v in eigenvectors],
            "B": vectors_info,
            "ladder": ladder_info,
            "cels_sizes": jlatex(cels_sizes),
            "cels": jcels,
            "transition_matrix": get_transition_matrix,
        }

    def process(self, mx):
        self.matrix = sp.Matrix(mx)
        self.clear_datapack()

        mx_sp = sp.Matrix(self.matrix)
        handmade_vectors = []

        eigvecs = mx_sp.eigenvects()
        for e in eigvecs:
            ev_data = self.processing_of_eigenvalues(e)
            self.mx_datapack["eigenvalues"].append(ev_data)
            handmade_vectors += ev_data["transition_matrix"]
            ev_data["transition_matrix"] = [jlatex(i) for i in ev_data["transition_matrix"]]

        P, J = mx_sp.jordan_form()

        if handmade_vectors:
            P_handmade = sp.Matrix.hstack(*handmade_vectors)  # столбцы
        else:
            P_handmade = sp.eye(self.matrix.rows)

        htm = jlatex(
            sp.Matrix(P_handmade).T
        )
        accuracy_of_calculations = check(self.matrix, P_handmade, J)

        self.mx_datapack["eigenvalues_count"] = len(eigvecs)
        self.mx_datapack["jordan_matrix"] = jlatex(J)
        self.mx_datapack["transition_matrix"] = jlatex(P)
        self.mx_datapack["handmade_transition_matrix"] = htm

        # флаг, что сошлись библиотечный и рукописный переходы
        self.mx_datapack["http"] = f"${accuracy_of_calculations}$"

    def get_datapack(self) -> str:
        return json.dumps(self.mx_datapack)


if __name__ == "__main__":
    s = Solver()
    s.process(MX)
    print(s.get_datapack())
