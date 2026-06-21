#!/usr/bin/env python3
import math, random
import sys
import numpy as np
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import print_tour, read_input

class TSPSolver:
    def __init__(self, cities, split_num):
        self.cities = cities
        self.split_num = split_num
        self.clusters = []     # 各クラスターに含まれる id の配列. 
        self.tours = []

        self.temperature_initial = 100
        self.temperature_limit = 0.001
        self.cooling_rate = 0.99999 
        self.temperature = self.temperature_initial

        self.dist_matrix = cdist(self.cities, self.cities)

    # ------------------------------------------------------------------
    # 1. クラスタリング O(NlogN * split_num^2)
    # ------------------------------------------------------------------
    
    def split_on_x_axis(self, city_id_list):
        sorted_city_id_list = sorted(city_id_list, key=lambda id: self.cities[id][0])
        return list(np.array_split(sorted_city_id_list, self.split_num))

    def split_on_y_axis(self, city_id_list):
        sorted_city_id_list = sorted(city_id_list, key=lambda id: self.cities[id][1])
        return list(np.array_split(sorted_city_id_list, self.split_num))

    def make_clusters(self):
        city_id_list = list(range(len(self.cities)))

        clusters_on_x_axis = self.split_on_x_axis(city_id_list)

        clusters_on_both_axis = []
        for cluster in clusters_on_x_axis:
            clusters_on_both_axis.extend(self.split_on_y_axis(cluster))

        self.clusters = clusters_on_both_axis
        return all([len(cluster) >= 4 for cluster in self.clusters])


    # ------------------------------------------------------------------
    # 距離・グラフの基本操作
    # ------------------------------------------------------------------
    def calc_dist_of_cities(self, city_id_1, city_id_2):
        return self.dist_matrix[city_id_1, city_id_2]
    
    # ------------------------------------------------------------------
    # 2. 各クラスタ内の部分巡回路作成 O(N^2)
    # ------------------------------------------------------------------
    def greedy(self, cluster_id):
        INF = 10**9
        visited = [False] * len(self.cities)

        cluster = self.clusters[cluster_id]
        root_id = cluster[0]
        cur_city_id = root_id
        visited[root_id] = True

        tour = [cur_city_id]

        while True:
            min_dist = INF
            next_city_id = -1

            for city_id in cluster:
                if visited[city_id]:
                    continue
                dist = self.calc_dist_of_cities(cur_city_id, city_id)
                if dist < min_dist:
                    min_dist = dist
                    next_city_id = city_id

            if next_city_id == -1:
                break

            tour.append(next_city_id)
            cur_city_id = next_city_id
            visited[cur_city_id] = True

        self.tours.append(tour)


    # ------------------------------------------------------------------
    # 3. 部分巡回路を 2-opt 法で改善 O(10^6)
    # ------------------------------------------------------------------

    ## path1: id1⇔id2, path2: id3⇔id4 を id1⇔id3, id2⇔id4 と交換した方が巡回路長が短くなる場合交換する. 
    ## id1⇔id4, id2⇔id3 と交換するとグラフが連結でなくなるので不可.
    def swap_path_if_shorter(self, tour, id_list):
        self.temperature *= self.cooling_rate

        tour_id1, tour_id2, tour_id3, tour_id4 = id_list
        city_id1, city_id2, city_id3, city_id4 = [tour[i] for i in id_list]

        cur_path_length = self.calc_dist_of_cities(city_id1, city_id2) + self.calc_dist_of_cities(city_id3, city_id4)
        new_path_length = self.calc_dist_of_cities(city_id1, city_id3) + self.calc_dist_of_cities(city_id2, city_id4)

        if cur_path_length <= new_path_length:
            random_num = random.uniform(0, 1)
            P = math.exp((cur_path_length - new_path_length) / self.temperature)
            if P < random_num:   
                return

        tour[tour_id2:tour_id3 + 1] = reversed(tour[tour_id2:tour_id3 + 1])

    def apply_2opt_with_sa(self, cluster_id):
        tour = self.tours[cluster_id]
        self.temperature = self.temperature_initial
        while self.temperature > self.temperature_limit:
            tour_id1 = random.randint(0, len(tour)-1)
            tour_id3 = random.randint(0, len(tour)-1)

            if tour_id1 > tour_id3:
                tour_id3, tour_id1 = tour_id1, tour_id3

            tour_id2 = (tour_id1 + 1) % len(tour)
            tour_id4 = (tour_id3 + 1) % len(tour)

            id_list = [tour_id1, tour_id2, tour_id3, tour_id4]
            if tour_id3 - tour_id1 < 2 or (tour_id1 == 0 and tour_id3 == len(tour) - 1):
                continue
            
            self.swap_path_if_shorter(tour, (tour_id1, tour_id2, tour_id3, tour_id4))
        
        self.tours[cluster_id] = tour


    # ------------------------------------------------------------------
    # 作成した部分巡回路の評価 O(N)
    # ------------------------------------------------------------------

    ## 指定のクラスターを巡回するパスを配列に書き出す. 
    def tour_to_path_tour(self, cluster_id):
        tour = self.tours[cluster_id]
        if tour is None:
            return []

        path_tour = []
        for i in range(len(tour)-1):
            path_tour.append((tour[i], tour[i+1]))
        path_tour.append((tour[len(tour)-1], tour[0]))

        return path_tour

    def rotate_tour(self, tour, start_city_id):
        start_tour_id = -1
        for tour_id in range(len(tour)):
            if tour[tour_id] == start_city_id:
                start_tour_id = tour_id
        
        tour[:] =  tour[start_tour_id:] + tour[:start_tour_id]

    # ------------------------------------------------------------------
    # 3. クラスタ同士の結合 O(N ^ 2 * split_num ^ 2)
    # ------------------------------------------------------------------
    ## path1 と path2（2つは別のクラスターのパス）を繋ぎ替えた場合に, 全体の巡回路の長さがどれくらい増えるか計算する. 増えるとき正の数を返す.
    ## また, path1: id1⇔id2, path2: id3⇔id4 を id1⇔id3, id2⇔id4 と繋ぎ替えても id1⇔id4, id2⇔id3 と繋ぎ替えてもよいので,
    ## id1⇔id3, id2⇔id4 とした方が全体の巡回路が短くなる場合 "1to3_and_2to4" を, 他方の場合は "1to4_and_2to3" を返す.
    def calc_decrease_dist_and_better_swapping_way(self, path1, path2):
        city_id1, city_id2 = path1
        city_id3, city_id4 = path2

        cur_dist = self.calc_dist_of_cities(city_id1, city_id2) + self.calc_dist_of_cities(city_id3, city_id4)
        dist_1to3_and_2to4 = self.calc_dist_of_cities(city_id1, city_id3) + self.calc_dist_of_cities(city_id2, city_id4)
        dist_1to4_and_2to3 = self.calc_dist_of_cities(city_id1, city_id4) + self.calc_dist_of_cities(city_id2, city_id3)

        if dist_1to3_and_2to4 < dist_1to4_and_2to3:
            return (dist_1to3_and_2to4 - cur_dist, "1to3_and_2to4")
        return (dist_1to4_and_2to3 - cur_dist, "1to4_and_2to3")

    def join_clusters(self):
        INF = 10**9

        while len(self.tours) > 1:
            best_choice = {
                'decrease_dist': INF,
                'way': None,
            }

            for path0 in self.tour_to_path_tour(0):
                for cluster_id in range(1, len(self.tours)):
                    for path1 in self.tour_to_path_tour(cluster_id):
                        decrease_dist, way = self.calc_decrease_dist_and_better_swapping_way(path0, path1)
                        if decrease_dist < best_choice['decrease_dist']:
                            best_choice = {
                                'decrease_dist': decrease_dist,
                                'path0': path0,
                                'path1': path1,
                                'cluster_id': cluster_id,
                                'way': way,
                            }

            if best_choice['way'] == "1to3_and_2to4":
                id1, id2 = best_choice['path0']
                id3, id4 = best_choice['path1']
                cluster_id = best_choice['cluster_id']
                self.rotate_tour(self.tours[0], id2)
                self.rotate_tour(self.tours[cluster_id], id4)
                self.tours[cluster_id].reverse()
                self.tours[0].extend(self.tours[cluster_id])
                self.tours[cluster_id] = None

            elif best_choice['way'] == "1to4_and_2to3":
                id1, id2 = best_choice['path0']
                id3, id4 = best_choice['path1']
                cluster_id = best_choice['cluster_id']
                self.rotate_tour(self.tours[0], id2)
                self.rotate_tour(self.tours[cluster_id], id4)
                self.tours[0].extend(self.tours[cluster_id])
                self.tours[cluster_id] = None
            else:
                break

        self.tours = [tour for tour in self.tours if tour is not None]
        self.clusters = [list(range(len(self.cities)))]

    # ------------------------------------------------------------------
    # 補助表示・評価 O(N)
    # ------------------------------------------------------------------
    def plot_graph(self, graph_name=""):
        for i in range(len(self.clusters)):
            city_id_tour = self.tours[i]

            x = [self.cities[i][0] for i in city_id_tour]
            y = [-self.cities[i][1] for i in city_id_tour]
            x.append(self.cities[city_id_tour[0]][0])
            y.append(-self.cities[city_id_tour[0]][1])

            plt.plot(x, y, marker='.', markersize=5)
            plt.scatter(x[0], y[0], c='red')
        plt.savefig(f'graph_{graph_name}.png')
        plt.close()

    ## 指定のクラスターの部分巡回路長を計算. 
    def calc_tour_length(self, cluster_id=0):
        path_tour = self.tour_to_path_tour(cluster_id)
        tour_length = 0
        for path in path_tour:
            tour_length += self.calc_dist_of_cities(path[0], path[1])
        return tour_length

    # ------------------------------------------------------------------
    # 5. 全体の流れ
    # ------------------------------------------------------------------
    def solve(self):
        succeed = self.make_clusters()
        if not succeed:
            return (False, [], None)
        
        for cluster_id in range(len(self.clusters)):
            self.greedy(cluster_id)
            self.apply_2opt_with_sa(cluster_id)
        self.plot_graph(f"greedy_{self.split_num}")

        self.join_clusters()
        self.plot_graph(f"joint_{self.split_num}")

        print(self.split_num,": ",self.calc_tour_length())

        return (True, self.tours[0], self.calc_tour_length())

def solve(cities):
    max_split_num = 5

    INF = 10**9
    best_tour = []
    best_tour_length = INF

    for i in range(1, max_split_num + 1):
        tsp_solver = TSPSolver(cities, i)
        succeed, tour, tour_length = tsp_solver.solve()
        if succeed and tour_length < best_tour_length:
            best_tour_length = tour_length
            best_tour = tour

    print("best", best_tour_length)

    return best_tour_length



if __name__ == '__main__':
    assert len(sys.argv) > 1
    tour = solve(read_input(sys.argv[1]))
    # print_tour(tour)
