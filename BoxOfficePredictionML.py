import pandas as pd
import numpy as np
import ast
import copy
import matplotlib.pyplot as plt

MOVIE_DATA = "./data/tmdb_5000_movies.csv"
TMP_DUMP = "./data/md.csv"

all_genre_list = []


def get_all_genres(movie_df):

    all_genre_list = []
    for genre_list in movie_df["genres"]:
        for g in genre_list:
            if g["name"] not in all_genre_list:
                all_genre_list.append(g["name"])
    return all_genre_list


def check_gtype(glist, g):

    for g_movie in glist:
        if g_movie["name"] == g:
            return 1
    return 0


def load_data():
    movie_df = pd.read_csv(MOVIE_DATA)
    movie_df = movie_df[movie_df.revenue >= 1000000]
    print(movie_df.describe)
    movie_df["genres"] = movie_df["genres"].apply(ast.literal_eval)

    all_genre_list = get_all_genres(movie_df)
    for g in all_genre_list:
        g_name = "is_" + g
        movie_df[g_name] = movie_df["genres"].apply(check_gtype, g=g)

    movie_df.to_csv(TMP_DUMP)

    XY_df = movie_df.filter(like="is_")
    # del XY_df["is_Foreign"]

    # del XY_df["is_Western"]
    # del XY_df["is_Documentary"]
    # del XY_df["is_Music"]
    # del XY_df["is_War"]
    # del XY_df["is_History"]
    # del XY_df["is_Mystery"]

    print(XY_df.columns)
    for c in XY_df.columns:
        print(c, XY_df[c].sum())
    XY_df["revenue"] = movie_df["revenue"] / 100000000
    XY = XY_df.to_numpy()
    Y = XY[:, -1]
    X = np.delete(XY, -1, axis=1)
    print(Y)
    print(XY.shape, X.shape, Y.shape)
    XY_df.to_csv(TMP_DUMP)

    X_train = X[1:]
    # X_test = X[3000:]
    Y_train = Y[1:]
    # Y_test = Y[3000:]
    return (X_train, Y_train, X_train, Y_train)


def compute_cost(X, Y, w, b):
    # fx_wb_i=wx+b
    # J=1/(2m) * i=1 to m[sum (fx_wb_i-y)**2]
    m, n = X.shape
    total_cost = 0.0
    print(m, n)
    for i in range(m):
        f_i = np.dot(X[i], w) + b
        total_cost += (f_i - Y[i]) ** 2
    total_cost /= 2
    total_cost /= m
    return total_cost


def compute_gradient(X, Y, w, b):
    # dj_db = 1/m *(fx-y)
    m, n = X.shape
    dj_db = 0
    dj_dw = np.zeros_like(w)
    for i in range(m):
        err_i = (np.dot(X[i], w) + b) - Y[i]
        dj_db += err_i
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_dw[j] /= m
    dj_db /= m
    return dj_db, dj_dw


def gradient_descent(
    X, y, w_in, b_in, cost_func, gradient_func, learning_rate, num_iters
):
    w = copy.deepcopy(w_in)
    b = b_in

    w_hist = [w]
    b_hist = [b]
    cost_hist = [cost_func(X, y, w, b)]
    p_hist = [[w, b]]

    for i in range(num_iters):
        dj_db, dj_dw = gradient_func(X, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        w_hist.append(w)
        b_hist.append(b)
        p_hist.append([w, b])
        if i % 1000 == 0:
            print(f"Iteration {i}, Cost: {cost_func(X, y, w, b)}")
    return (w, b, cost_hist)


def plot_data(X, Y, genre):
    fig, ax = plt.subplots(4, 5, figsize=(20, 3), sharey=True)
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].scatter(X[:, i * j], Y)
        # ax[i][j].set_xlabel(genre[i * j ])
        ax[i][0].set_ylabel("Revenue (hun mil)")
    plt.show()


def predict(X_test, w, b):
    return np.dot(X_test, w) + b


# X_train, Y_train, X_test, Y_test = load_data()
# w = np.zeros_like(X_train[0])
# b = 0
# # b = 0.07
# # w = [0.624, 1.16, 0.805, 0.519, 0.102, 0.279, 0.183, 1.331, 0.422, 0.219, 0.265, 0.081]
# # b = 0.14
# # w = [0.606, 1.156, 0.791, 0.499, 0.092, 0.241, 0.155, 1.315, 0.408, 0.183, 0.251, 0.081]
# # b = 0.31
# # w = [0.559, 1.122, 0.755, 0.447, 0.067, 0.140, 0.082, 1.274, 0.373, 0.087, 0.218, 0.081]
# # b = 0.45
# # w = [0.522, 1.095, 0.728, 0.406, 0.047, 0.061, 0.025, 1.241, 0.345, 0.012, 0.191, 0.081]

# # plot_data(X_train, Y_train, all_genre_list)
# cost = compute_cost(X_train, Y_train, w, b)
# tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, Y_train, w, b)
# print(f"cost at initial w,b: {cost}")
# print(f"dj_db at initial w,b: {tmp_dj_db}")
# print(f"dj_dw at initial w,b: \n {tmp_dj_dw}")
# iterations = 30001
# alpha = 5.0e-4
# # run gradient descent
# w_final, b_final, J_hist = gradient_descent(
#     X_train, Y_train, w, b, compute_cost, compute_gradient, alpha, iterations
# )
# print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
# cost = compute_cost(X_train, Y_train, w_final, b_final)
# print(f"cost at final w,b: {cost}")

# tt = len(Y_test)
# ct = 0
# th = 0.5
# yhatl = []
# for x, y in zip(X_test, Y_test):
#     yhat = predict(x, w_final, b_final)
#     yhatl.append(yhat)
#     if abs(y - yhat) < th:
#         ct += 1
#         # print(f"X:{x}, Predicted Y: {yhat}, Actual Y: {y}")
# print(f"Accuracy = {ct/tt:0.2f}")
# # plt.scatter(yhatl, Y_test)
# plt.scatter(X_test[:, 0], Y_test, c="r")

# plt.scatter(X_test[:, 0], yhatl)
# plt.show()
