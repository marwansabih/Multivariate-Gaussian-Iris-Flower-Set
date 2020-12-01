import pandas as pd
import scipy
import numpy as np

from scipy.stats import multivariate_normal


def measured_means(df):
    return df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].mean(axis=0).to_numpy()


def measured_cov(df):
    return df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].cov().to_numpy()


def measured_val(df):
    return df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()


def predict_set_vir_ver(values):
    pred = np.array([mvg_setosa.pdf(values),
                     mvg_virginica.pdf(values),
                     mvg_versicolor.pdf(values)])
    idxs= np.argmax(np.swapaxes(pred, 0, 1), axis=1)
    return np.array([to_one_hot(idx) for idx in idxs])


def to_one_hot(idx):
    vec = np.zeros(3)
    vec[idx] = 1.0
    return vec


def pred_err_percent(gt, pred):
    return 100 * (1 - np.sum(gt * pred) / gt.shape[0])


iris_df = pd.read_csv('archive/Iris.csv', sep=',', index_col=None, na_values=['NA'])

setosa_train = iris_df.loc[iris_df['Species'] == 'Iris-setosa'][0:40]
virginica_train = iris_df.loc[iris_df['Species'] == 'Iris-virginica'][0:40]
versicolor_train = iris_df.loc[iris_df['Species'] == 'Iris-versicolor'][0:40]

setosa_test = iris_df.loc[iris_df['Species'] == 'Iris-setosa'][40:50]
virginica_test = iris_df.loc[iris_df['Species'] == 'Iris-virginica'][40:50]
versicolor_test = iris_df.loc[iris_df['Species'] == 'Iris-versicolor'][40:50]

mean_setosa = measured_means(setosa_train)
mean_virginica = measured_means(virginica_train)
mean_versicolor = measured_means(versicolor_train)

cov_setosa = measured_cov(setosa_train)
cov_virginica = measured_cov(virginica_train)
cov_versicolor = measured_cov(versicolor_train)

print("Mean of setosa: ")
print(mean_setosa)
print("Mean of virgincia: ")
print(mean_virginica)
print("Mean of versicolor: ")
print(mean_versicolor)

print("\n")

print("Cov_Matrix of setosa: ")
print(cov_setosa)
print("Cov_Matrix of virginica: ")
print(cov_virginica)
print("Cov_Matrix of versicolor: ")
print(cov_versicolor)

mvg_setosa = multivariate_normal(mean_setosa, cov_setosa)
mvg_virginica = multivariate_normal(mean_virginica, cov_virginica)
mvg_versicolor = multivariate_normal(mean_versicolor, cov_versicolor)

test_set_setosa = measured_val(setosa_test)
test_set_virginica = measured_val(virginica_test)
test_set_versicolor = measured_val(versicolor_test)

setosa_pred = predict_set_vir_ver(test_set_setosa)
virginica_pred = predict_set_vir_ver(test_set_virginica)
versicolor_pred = predict_set_vir_ver(test_set_versicolor)


setosa_enc = np.array([1.0, 0.0, 0.0])
virginica_enc = np.array([0.0, 1.0, 0.0])
versicolor_enc = np.array([0.0, 0.0, 1.0])

setosa_gt = np.tile(setosa_enc, (10, 1))
virginica_gt = np.tile(virginica_enc, (10, 1))
versicolor_gt = np.tile(versicolor_enc, (10, 1))

setosa_error = pred_err_percent(setosa_gt, setosa_pred)
virginica_error = pred_err_percent(virginica_gt, virginica_pred)
versicolor_error = pred_err_percent(versicolor_gt, versicolor_pred)

print("\n")
print("Absolut miss classification error in percent for setosa: " + str(setosa_error) + "%")
print("Absolut miss classification error in percent for virginica: " + str(virginica_error) + "%")
print("Absolut miss classification error in percent for versicolor: " + str(versicolor_error) + "%")
