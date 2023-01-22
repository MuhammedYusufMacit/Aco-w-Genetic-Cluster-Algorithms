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

# FONKSİYONLAR
finishvariable=0.0
totalLength=0.0
number_of_iterations=1
colony_size=1
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
"""
genetic + En iyi rotayı bulduktan sonra, bir önceki rotaylar düğümleri birleştirip 
çaprazlama /mutasyonözelliği kullanılabilir
Kümeleme + 
"""
# karıncanın bir sonraki gideceğim düğüm belirlendi, burada tüm rota oluşturuluyor
def tour_construction():
    tour_nodes = [1] #[random.randint(0, number_of_nodes - 1)]  # rastgele gidilecek düğüm

    while len(tour_nodes) < number_of_nodes:
        ekle = _select_node(tour_nodes)
        tour_nodes.append(ekle)
    return tour_nodes
"""
genetic +
Kümeleme +
"""
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
"""
genetic +
Kümeleme -
"""
def _aco(nodes):
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
            ants_nodes.append(tour_construction())
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


def run(mode, nodes):
    print('Started : {0}'.format(mode))
    final_best_nodes, final_best_distance = _aco(nodes)

    print('Ended : {0}'.format(mode))
    print('Sequence : <- {0} ->'.format(' - '.join(str(labels[i]) for i in final_best_nodes)))
    print('Total distance travelled to complete the tour : {0}\n'.format(round(final_best_distance, 2)))
    run.totalLength=round(final_best_distance, 2)
    return final_best_nodes

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

def find_closest_distance_between_polys(poly0,poly1,poly2,poly3):
    #TODO Burada n tane çokgen gelecek, n çokgenin n^2-2(?) adet bağlantısı olacak
    #Bu bağlantılar arrayde belirli bir algoritma (Sıra) ile tutulmalıdır. 
    #Bu bağlantı nodeları (Bridge Nodes) Normal Nodeların bulunduğu kümelerden çıkarılmalı, Son durak olarak eklenebilir ?
    
    
    # find the closest points between the two polygons
    closest_points = nearest_points(poly0, poly1)
    
    point0to1 = np.array([closest_points[0].x, closest_points[0].y])
    point1to0 = np.array([closest_points[1].x, closest_points[1].y])
    
    closest_points = nearest_points(poly1, poly2)

    point1to2 = np.array([closest_points[0].x, closest_points[0].y])
    point2to1 = np.array([closest_points[1].x, closest_points[1].y])
    
    closest_points = nearest_points(poly2, poly3)

    point2to3 = np.array([closest_points[0].x, closest_points[0].y])
    point3to2 = np.array([closest_points[1].x, closest_points[1].y])
    
    closest_points = nearest_points(poly3, poly0)

    point3to0 = np.array([closest_points[0].x, closest_points[0].y])
    point0to3 = np.array([closest_points[1].x, closest_points[1].y])
    
    closest_points = nearest_points(poly0, poly2)

    point0to2 = np.array([closest_points[0].x, closest_points[0].y])
    point2to0 = np.array([closest_points[1].x, closest_points[1].y])
    
    closest_points = nearest_points(poly1, poly3)

    point1to3 = np.array([closest_points[0].x, closest_points[0].y])
    point3to1 = np.array([closest_points[1].x, closest_points[1].y])

    closest_points=[point0to1,point1to0,point1to2,point2to1,point2to3,point3to2,point3to0,point0to3,point0to2,point2to0,point1to3,point3to1]
    
    return closest_points

    
def pre4run(nodes):
    global pheromone
    global labels
    global cost_distance
    global number_of_nodes
    number_of_nodes=len(nodes)
    cost_distance = get_euclid_distance(initial_pheromone, number_of_nodes, nodes)
    pheromone = [[initial_pheromone] * len(nodes) for _ in range(len(nodes))]
    labels = range(1, number_of_nodes + 1)
    final_best_nodes = run(mode, nodes)
    plot(nodes, final_best_nodes, mode, labels)
    return final_best_nodes
    
if __name__ == '__main__':
    # DEĞİŞKENLER
    mode = 'Standard ACO without 2opt'
    nodes = pd.read_excel('C:\\Users\\Administrator\\Desktop\\Masaüstü\\Tez\\berlin52.xls').values
    number_of_nodes = len(nodes)
    global cost_distance
    global pheromone

    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(nodes)
 
    
    #TODO Merkez noktaları en yakın kümeler arasında geçiş yapılacak
    cluster_centers_=kmeans.cluster_centers_
    
    nodelar = np.empty((52,3))
    nodelar[:,:-1] = nodes
    nodelar[:,2]=kmeans.labels_
    nodesX0,nodesY0,nodesX1,nodesY1,nodesX2,nodesY2,nodesX3,nodesY3=np.empty([0,2]),np.empty([0,2]),np.empty([0,2]),np.empty([0,2]),np.empty([0,2]),np.empty([0,2]),np.empty([0,2]),np.empty([0,2])

    #TODO Otomatik hale getirilmeli, Küme geçişlerine Bridge Node, -1,-1 node'u gibi belirgin bir şey koyulabilir.
    #Burada yapılan bir bug Run fonksiyonunu etkiler.
    for i in range(len(nodelar)):
        node=nodelar[i]
        if node[2]==0:
            nodesX0 = np.append(nodesX0,node[0])
            nodesY0 = np.append(nodesY0,node[1])
        if node[2]==1:
            nodesX1 = np.append(nodesX1,node[0])
            nodesY1 = np.append(nodesY1,node[1])
        if node[2]==2:
            nodesX2 = np.append(nodesX2,node[0])
            nodesY2 = np.append(nodesY2,node[1])
        if node[2]==3:
            nodesX3 = np.append(nodesX3,node[0])
            nodesY3 = np.append(nodesY3,node[1])
            
    nodes0=np.column_stack((nodesX0,nodesY0))
    nodes1=np.column_stack((nodesX1,nodesY1))
    nodes2=np.column_stack((nodesX2,nodesY2))
    nodes3=np.column_stack((nodesX3,nodesY3))
    
    #Çokgenler Arasındaki en kısa yol için
    poly0 = Polygon(nodes0)
    poly1 = Polygon(nodes1)
    poly2 = Polygon(nodes2)
    poly3 = Polygon(nodes3)

    closest_points = find_closest_distance_between_polys(poly0, poly1, poly2, poly3)
    
    rho = 0.5
    number_of_iterations=100
    colony_size=100
    initial_pheromone = 1.0
    initial_pheromone_weight = 1.0
 

    nodes00=pre4run(nodes0)
    nodes01=pre4run(nodes1)
    nodes02=pre4run(nodes2)
    nodes03=pre4run(nodes3)
    
    #TODO Tüm Nodelar SIRAYLA birleştirilmeli.
    # Her küme kendi arasında Tek bir arrayde run edebilirse bu işleme gerek kalmaz
    # Küme Nodeları ve Bridge Nodeları Bir arada tutulmuş olunur.
    finalnodes = np.empty([0,2])
    for node in nodes00:
        finalnodes=np.append(finalnodes, nodes0[node-1])
    finalnodes=np.append(finalnodes, closest_points[0])
    finalnodes=np.append(finalnodes, closest_points[1])
    for node in nodes01:
        finalnodes=np.append(finalnodes, nodes1[node-1])
    finalnodes=np.append(finalnodes, closest_points[2])
    finalnodes=np.append(finalnodes, closest_points[3])
    for node in nodes02:
        finalnodes=np.append(finalnodes, nodes2[node-1])
    


    pre4run(nodes)
    #nodes0.extend(nodes1)
    
  
    
 #BEST 7542   


    
    
    
  
