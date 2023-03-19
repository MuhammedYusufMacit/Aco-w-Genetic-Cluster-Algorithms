import math
import random
import pandas as pd
from matplotlib import pyplot as plt
import time
import sympy #çokgenler arasında en kısa iki nokta
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
from shapely.ops import nearest_points
from shapely.geometry import Polygon

"""
TODO's
Cluster_End_Nodes Karınca Kolonisinın çalıştırılacağı nodeların bulunduğu kümelerden çıkarılmalı, Final_nodes kısmında eklenmeli
Cluster_start_Nodes her kümede tespit edilmeli, tour_construction fonksiyonuna gönderilmeli.
    
Genetic Algoritmalar için: En iyi rotayı bulduktan sonra, bir önceki rotalar düğümleri birleştirip çaprazlama /mutasyonözelliği kullanılabilir
Nodes Mapping N küme için yapılmalı
find_first_node fonksiyonundaki hatalar giderilmeli
Merkez noktaları en yakın kümeler arasında geçiş yapılabilir
Başlangıç ve bitiş nodeları elimizde var fakat bulduğumuz bu nodeları elimizdeki node'larla eşleyemiyoruz. En yakın nokta bulma işleminde bu sorun çözülebilir. Elimizdeki float değer olmamasına rağmen en yakın nokta bulan fonksiyon bir float değer döndürüyor.
"""

# FONKSİYONLAR
# weighted_random_choice()
# _select_node()
# tour_construction()
# get_instant_distance()
# get_euclid_distance()
# _add_pheromone()
# _evaporation()
# _aco()
# run()
# plot()
# find_closest_distance_between_polys()
# preprocess4run()

class Bridge_Node:

    def __init__(self, a, d, x_y): #constructor that takes 3 arguments (the self is the class itself)
        self.arrival_cluster=a
        self.departure_cluster=d
        self.x_y=x_y


# DEĞİŞKENLER
paths= 'C:\\Users\\TRON PCH\\Documents\\berlin52.xls', 'C:\\Users\\Turtle\\Datasets\\berlin52.xls', 'C:\\Users\\LENOVO\\OneDrive\\Masaüstü\\berlin52.xls'
path = paths[2]
n_clusters = 4
finishvariable=0.0
totalLength=0.0
rho = 0.5
number_of_iterations=300
colony_size=200
initial_pheromone = 1.0
initial_pheromone_weight = 1.0
cluster_end_points = []
cluster_start_points = []
def weighted_random_choice(choices):
    max = sum(choices.values())
    pick = random.uniform(0, max)
    current = 0
    for key, value in choices.items():
        current += value
        if current > pick:
            return key

# karıncanın gideceği sıraki düğüm belirleniyor
def _select_node(tour_nodes):
    alpha = 1.0
    beta = 5.0

    # ziyaret edilmiş ve edilmemiş düğümleri belirlemek için
    unvisited_nodes = []
    for node in range(number_of_nodes):
        if node not in tour_nodes:
            unvisited_nodes.append(node)

    # düğüm seçimi için olasılık formülünün paydasını hesaplıyoruz
    probability_denominator = 0.0
    for unvisited_node in unvisited_nodes:
        probability_denominator += (1 / pow(cost_distance[tour_nodes[-1]][unvisited_node], beta)) * \
                                    pow(pheromone[tour_nodes[-1]][unvisited_node], alpha)


    # tüm düğümler için olasılık hesabı yaptık
    probabilities = {}
    for unvisited_node in unvisited_nodes:
        try:
            probabilities[unvisited_node] = pow(pheromone[tour_nodes[-1]][unvisited_node],
                                                alpha) * \
                                            (1 / pow(cost_distance[tour_nodes[-1]][unvisited_node],
                                                        beta)) / probability_denominator
        except ValueError:
            pass  # do nothing

    # en yüksek olasılığa göre gidilecek düğüm seçildi, rulet secimi
    selected = weighted_random_choice(probabilities)

    return selected

# karıncanın bir sonraki gideceğim düğüm belirlendi, burada tüm rota oluşturuluyor
def tour_construction(first_node):
    if first_node==-1:
        #print("Başlangıç Node'u Null geldi, değer 0 olarak atandı")
        first_node=[[0]]
    
    tour_nodes = [first_node[0][0]]
    
  
    while len(tour_nodes) < number_of_nodes:
        ekle = _select_node(tour_nodes)
        tour_nodes.append(ekle)
    return tour_nodes

# tüm rotası belirlenen karıncanın yolunun ne kadar olduğu belirleniyor
def get_instant_distance(tour_nodes):
    # karıncanın o ana kadar dolaştığı yolun uzunluğunu belirlemek için kullanılıyor
    distance = 0.0
    for i in range(number_of_nodes):
        distance = distance + cost_distance[tour_nodes[i]][
            tour_nodes[(i + 1) % number_of_nodes]]

    return distance


# euclid uzaklığı hesaplandı
def get_euclid_distance(initial_pheromone, number_of_nodes, nodes):
    # uzaklık hesabı yapıp self.cost içerisine kayıt edeceğim için ilk değer atamasına ihtiyacım var
    cost = [[None] * number_of_nodes for _ in range(number_of_nodes)]

    for i in range(number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            # cost için her nokta arasındaki euclid hesabı yapılıyor
            cost[i][j] = cost[j][i] = math.sqrt(
                pow(nodes[i][0] - nodes[j][0], 2.0) + pow(nodes[i][1] - nodes[j][1], 2.0))

    return cost

def _add_pheromone(tour_nodes, distance, cost=1.0):
    delta = initial_pheromone_weight / distance  # delta = 1/C^s formülünden
    for i in range(number_of_nodes):
        pheromone[tour_nodes[i]][tour_nodes[(i + 1) % number_of_nodes]] += cost * delta

    return pheromone

# buharlaşma için
def _evaporation():
    for i in range(number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            pheromone[i][j] *= (1.0 - rho)

    return pheromone

def _aco(nodes,first_node):
    # son durumda en iyi düğüm ve uzunluk bilgisini tutmak için kullanılacaklar

    final_best_nodes = None
    final_best_distance = float("inf")

    ants_nodes = [] # [0] * colony_size
    ants_distance = [] # [0] * colony_size

    start = time.perf_counter()

    for step in range(number_of_iterations):
        
        # start1 = time.perf_counter()
        # her iterasyonda karıncalar için rota oluşturulup yol hesabı yapılıyor
        for i in range(colony_size):
            ants_nodes.append(tour_construction(first_node))
        # finish1 = time.perf_counter()
        # print(f'Finished in {round(finish1 - start1, 4)} sec(s)')

        for ant in ants_nodes:
            ants_distance.append(get_instant_distance(ant))

        # oluşturulan rotalardan hangisinin daha iyi olduğu kontrol ediliyor
        for i in range(colony_size):
            pheromone = _add_pheromone(ants_nodes[i], ants_distance[i])
            if ants_distance[i] < final_best_distance:
                final_best_nodes = ants_nodes[i]
                final_best_distance = ants_distance[i]
                
        # işlem sonucu buharlaşma yapılıyor
        pheromone = _evaporation()
        #plot(nodes, final_best_nodes, mode, labels) #!!!!!    
    finish = time.perf_counter()
    print()
    _aco.finishvariable=round(finish - start, 4)
    print(f'Finished in {round(finish - start, 4)} sec(s)')

    return final_best_nodes, final_best_distance


def run(mode, nodes,first_node):
    print('Started : {0}'.format(mode))
    final_best_nodes, final_best_distance = _aco(nodes,first_node)

    print('Ended : {0}'.format(mode))
    print('Sequence : <- {0} ->'.format(' - '.join(str(labels[i]) for i in final_best_nodes)))
    print('Total distance travelled to complete the tour : {0}\n'.format(round(final_best_distance, 2)))
    run.totalLength=round(final_best_distance, 2)
    return final_best_nodes

def finalplot(nodes,final_best_nodes):
    
    number_of_nodes=len(nodes)
    labels = range(1, number_of_nodes + 1)
    final_best_nodes = final_best_nodes
    final_best_distance = float("inf")
    #SUM(Kümelerin Distance'ı + köprülerin distance'ı)
    
    run.totalLength=round(final_best_distance, 2)
    print('Sequence : <- {0} ->'.format(' - '.join(str(labels[i]) for i in final_best_nodes)))
    print('Total distance travelled to complete the tour : {0}\n'.format(round(final_best_distance, 2)))
    plot(nodes, final_best_nodes, mode, labels)


# ekrana grafik çıkarmak için
def plot(nodes, final_best_nodes, mode, labels, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
    x = [nodes[i][0] for i in final_best_nodes]
    x.append(x[0])
    y = [nodes[i][1] for i in final_best_nodes]
    y.append(y[0])
    plt.plot(x, y, linewidth=line_width)
    plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
    printvariable="ACO TIME: " + str(_aco.finishvariable) +" ROUTE: "+ str(run.totalLength)+" Iter:" + str(number_of_iterations) + " CSize: " + str(colony_size)
    plt.title(printvariable)
    for i in final_best_nodes:
        plt.annotate(labels[i], nodes[i], size=annotation_size)
    if save:
        if name is None:
            name = '{0}.png'.format(mode) 
        plt.savefig(name, dpi=dpi)
    plt.show()
    plt.gcf().clear()


def closest_points(matrix1, matrix2):
    min_distance = math.inf
    closest_points = None
 

    for point1 in matrix1:
        for point2 in matrix2:
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_points = (point1, point2)

    return closest_points

def find_closest_distance_between_polys(polygons):

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            nearestpoints = closest_points(polygons[i], polygons[j])
            point_i_to_j=Bridge_Node(i, j,[nearestpoints[0]])
            
            nearestpoints = closest_points(polygons[j], polygons[i])
            point_j_to_i=Bridge_Node(j, i,[nearestpoints[0]])
            
            if j-i ==1 or (i==n_clusters-1 and j==0):
                cluster_end_points.append(point_i_to_j)
                cluster_start_points.append(point_j_to_i)
                
    

    return nearestpoints,cluster_end_points,cluster_start_points

    
def preprocess4run(nodes,first_node):
    global pheromone
    global labels
    global cost_distance
    global number_of_nodes
    number_of_nodes=len(nodes)
    cost_distance = get_euclid_distance(initial_pheromone, number_of_nodes, nodes)
    pheromone = [[initial_pheromone] * len(nodes) for _ in range(len(nodes))]
    labels = range(1, number_of_nodes + 1)
    first_node=find_first_node(nodes,first_node)
    final_best_nodes = run(mode, nodes,first_node)
    plot(nodes, final_best_nodes, mode, labels)
    return final_best_nodes

def find_first_node(nodes,first_node):
    index=-1
    index = np.argwhere((nodes == first_node.x_y).all(axis=1))
    
    print("FIRST NODE IN CLUSTER :",index)
    print(nodes[index])

    return index

def remove_end_points(nodes,cluster_end_point):
    index = np.argwhere((nodes == cluster_end_point.x_y).all(axis=1))
    nodes = np.delete(nodes, index, axis=0)
    
    return nodes
    
    
def final_nodes_concatinating():
    
    bridge_nodes = np.empty([2,0])
    for i in range(n_clusters,1):
        for point in cluster_end_points:
            if point.arrival_cluster==i-1 and point.departure_cluster==i:
                np.append(bridge_nodes, point)
    
    
    finalnodes = np.concatenate((
    nodes[0][nodes00-1],
    [cluster_end_points[0].x_y[0]],
    nodes[1][nodes01-1],                           
    [cluster_end_points[1].x_y[0]],
    nodes[2][nodes02-1],
    [cluster_end_points[2].x_y[0]],
    nodes[3][nodes03-1],
    [cluster_end_points[3].x_y[0]]
    ))
    return finalnodes



if __name__ == '__main__':

    start = time.perf_counter()
    mode = 'Standard ACO without 2opt'
    
    #kroa100 - pr1002 - berlin52
    nodes_excel = pd.read_excel(path).values
    number_of_nodes = len(nodes_excel)
    global cost_distance
    global pheromone

    
    kmeans = KMeans(n_clusters, init='k-means++', random_state=0).fit(nodes_excel)

    cluster_centers_=kmeans.cluster_centers_
    #preprocess4run(cluster_centers_,Bridge_Node(-1, -1, -1))
    
    clustered_nodes = np.empty((len(nodes_excel),3))
    clustered_nodes[:,:-1] = nodes_excel
    clustered_nodes[:,2]=kmeans.labels_
    
    nodes = {i: np.empty([0,2]) for i in range(n_clusters)}

    # her node'da loop yapılır ve o node uygun sınıfa eklenir 
    for node in clustered_nodes:
        nodes[node[2]] = np.append(nodes[node[2]], [node[:2]], axis=0)

    # her sınıf için poligon oluşturulur. 
    polys = {i: Polygon(nodes[i]) for i in range(n_clusters)}

    
    nearest_points_,cluster_end_points,cluster_start_points = find_closest_distance_between_polys(nodes)
    
        
    nodes = [remove_end_points(nodes[i],cluster_end_points[i]) for i in range(n_clusters)]


      
    # TODO : kümenin gezi için başlayacağı node preprocess4run a ek parametre eklenerek yollanmalı. ?(nur)
    #preprocessed_nodes = [preprocess4run(nodes[i],cluster_start_points[i].x_y ) for i in range(n_clusters)] ?(nur)

    preprocessed_nodes = [preprocess4run(nodes[i],cluster_start_points[(i-1)%n_clusters]) for i in range(n_clusters)]
    
    #Todo Mapping N küme için yapılmalı
    nodes00, nodes01, nodes02, nodes03 = map(np.array, preprocessed_nodes) 
    
    finalnodes = np.empty([2,0])
    finalnodes = final_nodes_concatinating()
    
 
    nodes=finalnodes.reshape(len(clustered_nodes),2)
    final_best_nodes=[]
    for i in range(len(nodes)):
        final_best_nodes.append(i)
        
    finalplot(nodes,final_best_nodes)
    

    finish = time.perf_counter()
    print(f'Toplam Süre {round(finish - start, 2)} saniye.')
