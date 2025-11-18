import numpy as np
import sympy as sp
import itertools


def to_list(mx: np.ndarray) -> list:
    return [[int(j) for j in i] for i in mx.tolist()]  # Преобразует в список


def searching_Jordan_cell_vectors(base_mx: np.ndarray, target):
    b_pow = 1
    vectors_count = 0
    proper_values = [[]]
    gluing_values = []
    printable = []

    while 1:
        mx = np.linalg.matrix_power(base_mx, b_pow)
        mx_rang = np.linalg.matrix_rank(mx)
        ev = sp.Matrix(mx).nullspace()
        pv = [tuple(v) for v in ev]
        printable.append(
            {
                "B": b_pow,
                "matrix": jlatex(sp.Matrix(np.array(mx, dtype=np.int32))),
                "vectors": [],
            }
        )

        for v in pv:
            v = [vv.p for vv in v]
            vectors_set = gluing_values + [v]
            matrix = np.array(vectors_set, dtype=np.int32)

            if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
                proper_values[-1] += [np.array(v, dtype=np.int32)]
                printable[-1]["vectors"] += [jlatex(v)]
                gluing_values += [v]
                vectors_count += 1
                if target == vectors_count:
                    break

        b_pow += 1
        next_mx = np.linalg.matrix_power(base_mx, b_pow)
        if mx_rang == np.linalg.matrix_rank(next_mx):
            break

        proper_values += [[]]

    if not proper_values[-1]:
        proper_values = proper_values[:-1]

    return proper_values, printable


def is_linearly_independent(vectors):
    matrix = np.array(vectors, dtype=np.int32)
    return np.linalg.matrix_rank(matrix) == matrix.shape[0]


def find_linearly_independent(possible_vectors, available_vectors) -> np.ndarray | None:
    ln = [i for i in available_vectors if not (i is None)]
    for v in possible_vectors:
        vectors_set = ln + [v]
        if is_linearly_independent(vectors_set):
            return v
    return None


def lower_ladder(high, x, b, l):
    for j in range(high - 1, -1, -1):
        upper_vector = l[j + 1][x].reshape(-1, 1)
        l[j][x] = (b @ upper_vector).reshape(-1)


def generate_vector(layer_vectrs, b, vector_len):
    values_range = range(-20, 21)

    for v in itertools.product(values_range, repeat=vector_len):
        v = np.array(v, dtype=np.int32)
        if np.all(b @ v.reshape(-1, 1) == 0):
            mx = layer_vectrs + [v]
            if is_linearly_independent(mx):
                return v
    return np.zeros(vector_len, dtype=np.int32)


def generate_ladder(b, label_vectors):
    l = [[None for _ in range(len(i))] for i in label_vectors]
    high = len(label_vectors) - 1

    for i in range(len(label_vectors[-1])):
        l[-1][i] = label_vectors[-1][i]
        lower_ladder(high, i, b, l)

    for i in range(high - 1, -1, -1):
        for j in range(len(label_vectors[i])):
            if l[i][j] is None:
                vector = find_linearly_independent(label_vectors[i], l[i])
                l[i][j] = (
                    vector
                    if not (vector is None)
                    else generate_vector(l[i], b, len(b[0][0]))
                )
                lower_ladder(i, j, b, l)

    ld = [[[int(v) for v in j] for j in i] for i in l]
    pld = [[jlatex(j) for j in i] for i in l[::-1]]
    return ld, tuple(pld)


def generate_transactions(ladder):
    lens_mx = np.zeros((len(ladder), len(ladder[0])), dtype=bool)
    for i in range(len(ladder)):
        lens_mx[i, : len(ladder[i])] = True
    lens_vector = np.array(np.sum(lens_mx, axis=0), dtype=np.int32)

    vectors = []
    for i in range(len(ladder[0])):
        for j in range(lens_vector[i]):
            vectors.append(ladder[j][i])

    return vectors, lens_vector


def check(a, p, j):
    # p^-1 * a * p
    p = np.array(p, dtype=np.int32).T
    p_inv = np.linalg.inv(p)
    j_test = (p_inv @ a @ p).astype(np.int32)
    return np.allclose(j_test, j)


def jlatex(obj):
    return "$" + sp.latex(obj) + "$"


def generate_cels(proper_value, cels_sizes):
    res = []
    for cs in cels_sizes:
        mx = np.eye(cs, dtype=np.int32) * proper_value
        for i in range(mx.shape[0]):
            if i + 1 < cs:
                mx[i][i + 1] = 1
            else:
                break
        res.append(jlatex(sp.Matrix(mx)))

    return res

