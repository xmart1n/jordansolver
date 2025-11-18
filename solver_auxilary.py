import numpy as np
import sympy as sp


def to_list(mx) -> list:
    # Преобразует в список
    sz = mx.shape
    l = np.array(mx).reshape(sz[1], sz[0]).tolist()
    return [[j.p for j in i] for i in l]


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
        n_info = {"B": b_pow, "matrix": mx.tolist(), "vectors": []}
        printable.append(n_info)

        for v in pv:
            v = [vv.p for vv in v]
            vectors_set = sum(proper_values, []) + [v]
            matrix = np.array(vectors_set, dtype=np.int32)

            if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
                proper_values[-1] += [v]
                gluing_values += [v]
                printable[-1]["vectors"] += [v]
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

                row = []
                for v in label_vectors[i]:
                    for e in l[i]:
                        if e and v != e and tuple([-vv for vv in v]) != e:
                            row.append(v)

                if row:
                    l[i][j] = row[0]
                    for k in range(i - 1, -1, -1):
                        vector = np.array(l[k + 1][i], dtype=np.int32).reshape(-1, 1)
                        l[k][i] = tuple((b @ vector).reshape(-1))

    return [[[int(v) for v in j] for j in i] for i in l]


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


def check(a, p, j):
    # p^-1 * a * p
    p = np.array(p, dtype=np.int32).T
    p_inv = np.linalg.inv(p)
    j_test = (p_inv @ a @ p).astype(np.int32)
    return np.all(j_test == j)
