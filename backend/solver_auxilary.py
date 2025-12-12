import sympy as sp
import itertools


def searching_Jordan_cell_vectors(base_mx: sp.Matrix, target):
    b_pow = 1
    vectors_count = 0
    proper_values = [[]]
    gluing_values = []
    printable = []

    while 1:
        mx = base_mx ** b_pow
        mx_rang = mx.rank()
        ev = mx.nullspace()
        printable.append(
            {
                "B": b_pow,
                "matrix": jlatex(mx),
                "vectors": [],
            }
        )

        for v in ev:
            vectors_set = gluing_values + [v]

            if is_linearly_independent(vectors_set):
                proper_values[-1].append(v)
                printable[-1]["vectors"].append(jlatex(v))
                gluing_values.append(v)
                vectors_count += 1
                if target == vectors_count:
                    break

        b_pow += 1
        next_mx = base_mx ** b_pow
        if mx_rang == next_mx.rank():
            break

        proper_values.append([])

    if not proper_values[-1]:
        proper_values = proper_values[:-1]

    return proper_values, printable


def is_linearly_independent(vectors):
    vectors = [sp.Matrix(v) for v in vectors]
    if not vectors:
        return True
    M = sp.Matrix.hstack(*vectors)
    return M.rank() == M.cols


def find_linearly_independent(possible_vectors, available_vectors) -> sp.Matrix | None:
    ln = [i for i in available_vectors if not (i is None)]
    for v in possible_vectors:
        vectors_set = ln + [v]
        if is_linearly_independent(vectors_set):
            return v
    return None


def lower_ladder(high, x, b, l):
    b = sp.Matrix(b)
    for j in range(high - 1, -1, -1):
        upper_vector = sp.Matrix(l[j + 1][x])
        l[j][x] = sp.Matrix(b * upper_vector)


def generate_vector(layer_vectrs, b, vector_len):
    b = sp.Matrix(b)
    values_range = range(-20, 21)

    existing = [v for v in layer_vectrs if v is not None]

    for v in itertools.product(values_range, repeat=vector_len):
        v_col = sp.Matrix(v)
        if (b * v_col).is_zero_matrix:
            mx = existing + [v_col]
            if is_linearly_independent(mx):
                return v_col

    return sp.zeros(vector_len, 1)


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
                    if vector is not None
                    else generate_vector(l[i], b, b.rows)
                )
                lower_ladder(i, j, b, l)

    pld = [[jlatex(j) for j in row] for row in l[::-1]]
    return l, tuple(pld)


def generate_transactions(ladder):
    cols = len(ladder[0])
    lens_vector = [
        sum(1 for row in ladder if len(row) > col)
        for col in range(cols)
    ]

    vectors = []
    for i in range(cols):
        for j in range(lens_vector[i]):
            vectors.append(ladder[j][i])

    return vectors, lens_vector


def check(a, p, j):
    A = sp.Matrix(a)
    P = sp.Matrix(p)
    J_expected = sp.Matrix(j)
    J_test = P.inv() * A * P
    return J_test.equals(J_expected)


def jlatex(obj):
    return "$" + sp.latex(obj) + "$"


def generate_cels(proper_value, cels_sizes):
    res = []
    for cs in cels_sizes:
        mx = sp.eye(cs) * proper_value
        mx = sp.Matrix(mx)

        for i in range(cs - 1):
            mx[i, i + 1] = 1

        res.append(jlatex(mx))

    return res
