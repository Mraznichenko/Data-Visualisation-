import random

import numpy as np
import pandas as pd

from typing import List, Tuple
from bokeh.models import ColumnDataSource, Slider, Div, Select, CustomJS
from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.palettes import Spectral10
from bokeh.transform import factor_cmap
from bokeh.io import show

# Use these centroids in the first iteration of you algorithm if "Random Centroids" is set to False in the Dashboard
DEFAULT_CENTROIDS = np.array([[5.664705882352942, 3.0352941176470587, 3.3352941176470585, 1.0176470588235293],
                              [5.446153846153847, 3.2538461538461543, 2.9538461538461536, 0.8846153846153846],
                              [5.906666666666667, 2.933333333333333, 4.1000000000000005, 1.3866666666666667],
                              [5.992307692307692, 3.0230769230769234, 4.076923076923077, 1.3461538461538463],
                              [5.747619047619048, 3.0714285714285716, 3.6238095238095243, 1.1380952380952383],
                              [6.161538461538462, 3.030769230769231, 4.484615384615385, 1.5307692307692309],
                              [6.294117647058823, 2.9764705882352938, 4.494117647058823, 1.4],
                              [5.853846153846154, 3.215384615384615, 3.730769230769231, 1.2076923076923078],
                              [5.52857142857143, 3.142857142857143, 3.107142857142857, 1.007142857142857],
                              [5.828571428571429, 2.9357142857142855, 3.664285714285714, 1.1]])


def get_closest(data_point: np.ndarray, centroids: np.ndarray)  -> Tuple[np.ndarray, int]:
    n = 0
    dist = np.linalg.norm(data_point - centroids[0])
    for i in range(centroids.shape[0]):
        dist2 = np.linalg.norm(data_point - centroids[i])
        if dist2 < dist:
            dist = dist2
            n = i
    return n


def k_means(data: np.ndarray, k: int = 3, n_iter: int = 500, random_initialization='False'):
    clustering = np.ndarray(shape=(k, 4))
    j = 0
    if random_initialization == 'True':
        while j < k:
            clustering[j][0] = random.uniform(5.4, 6.7)
            clustering[j][1] = random.uniform(2.8, 3.3)
            clustering[j][2] = random.uniform(2.9, 4.5)
            clustering[j][3] = random.uniform(0.8, 1.6)
            j += 1
    else:
        while j < k:
            clustering[j][0] = DEFAULT_CENTROIDS[j][0]
            clustering[j][1] = DEFAULT_CENTROIDS[j][1]
            clustering[j][2] = DEFAULT_CENTROIDS[j][2]
            clustering[j][3] = DEFAULT_CENTROIDS[j][3]
            j += 1
    data_array = np.zeros(shape=(k, data.shape[0], 4))
    counter = 0
    while counter < n_iter:
        old_clustering = clustering.copy()
        i = 0
        while i < 150:
            l = get_closest(np.array(
                [data['sepal_length'][i], data['sepal_width'][i], data['petal_length'][i], data['petal_width'][i]]),
                            clustering)
            data_array[l][i] = np.array(
                [data['sepal_length'][i], data['sepal_width'][i], data['petal_length'][i], data['petal_width'][i]])
            i += 1
        u = 0
        k_mean = np.array([0, 0, 0, 0])
        j = 0
        flag = False
        while u < k:
            for i in range(len(data_array[u])):
                if (data_array[u][i][0]) == 0:
                    j += 1
                if j == i:
                    u += 1
                    j = 0
                    if u == k:
                        flag = True
                        break
                    continue
                k_mean[0] = k_mean[0] + (data_array[u][i][0])
                k_mean[1] = k_mean[1] + (data_array[u][i][1])
                k_mean[2] = k_mean[2] + (data_array[u][i][2])
                k_mean[3] = k_mean[3] + (data_array[u][i][3])
            if flag:
                break
            clustering[u][0] = k_mean[0] / (i - j)
            clustering[u][1] = k_mean[1] / (i - j)
            clustering[u][2] = k_mean[2] / (i - j)
            clustering[u][3] = k_mean[3] / (i - j)
            j = 0
            u += 1

        counter += 1
        if np.isclose(old_clustering, clustering).all():
            break
        pass
    return data_array, counter


num_of_clusters = 3

def callback(attr, old, new):
    randomcentr = select.value
    build_everything(datapl, slider.value, randomcentr)



data: pd.DataFrame = flowers.copy(deep=True)
data = data.drop(['species'], axis=1)
datapl = data.copy()

# Create the dashboard
# 1. A Select widget to choose between random initialization or using the DEFAULT_CENTROIDS on top
# 2. A Slider to choose a k between 2 and 10 (k being the number of clusters)
# 4. Connect both widgets to the callback
# 3. A ColumnDataSource to hold the data and the color of each point you need
# 4. Two plots displaying the dataset based on the following table, have a look at the images
# in the handout if this confuses you.
#
#       Axis/Plot	Plot1 	Plot2
#       X	Petal length 	Petal width
#       Y	Sepal length	Petal length
#
# Use a categorical color mapping, such as Spectral10, have a look at this section of the bokeh docs:
# https://docs.bokeh.org/en/latest/docs/user_guide/categorical.html#filling
# 5. A Div displaying the currently number of iterations it took the algorithm to update the plot.
def build_everything(datapl, num_of_clusters, randomcentr):
    curdoc().clear()
    datapl2, counter = k_means(datapl, k=num_of_clusters, random_initialization=randomcentr)
    colors = ['green', 'yellow', 'blue', 'red', 'purple', 'orange', 'black', 'brown', 'pink', 'grey']
    s1 = figure(width=500, height=500, background_fill_color="#fafafa")
    for i in range(datapl2.shape[0]):
        datapl3 = []
        datapl4 = []
        for j in range(150):
            if datapl2[i][j][2] == 0:
                continue
            datapl3.append(datapl2[i][j][2])
            datapl4.append(datapl2[i][j][3])
        s1.circle(datapl3, datapl4, size=10, color=colors[i], alpha=0.8)
    s2 = figure(width=500, height=500, background_fill_color="#fafafa")
    for i in range(datapl2.shape[0]):
        datapl3 = []
        datapl4 = []
        for j in range(150):
            if datapl2[i][j][2] == 0:
                continue
            datapl3.append(datapl2[i][j][0])
            datapl4.append(datapl2[i][j][2])
        s2.circle(datapl3, datapl4, size=10, color=colors[i], alpha=0.8)
    div = Div(text="Number of iterations: " + str(counter), width=200, height=100)
    lt1 = column(select, slider, div)
    lt = row(lt1, s1, s2)
    curdoc().add_root(lt)
    curdoc().title = "DVA_ex_3"
slider = Slider(start=2, end=10, value=3, step=1, title="k")
slider.on_change('value', callback)
select = Select(title="Random Centroids:", value="False", options=['False', 'True'])
select.on_change("value", callback)
build_everything(datapl, num_of_clusters, randomcentr='False')


# if it don't start from 1st time, try again, its cause of computer memory work with big numbers:)
# Run it with
# bokeh serve --show dva_ex3_Lev_Maznichenko.py
