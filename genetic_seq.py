import random
import numpy as np
from collections import defaultdict
from dijkstra import Dijkstra
from vs.constants import VS

class GASequencer:
    def __init__(self, victims_cluster, map_obj, cost_line=1.0, cost_diag=1.5,
                 pop_size=100, max_generations=200, mut_rate=0.2,
                 time_limit=300.0, first_aid_time=5.0):
        """
        Inicializa o sequenciador GA.
        :param victims_cluster: dict {id: (pos, vitals)}
        :param map_obj: mapa para pathfinding
        """
        self.victims = victims_cluster
        self.map = map_obj
        self.cost_line = cost_line
        self.cost_diag = cost_diag
        self.population_size = pop_size
        self.max_gens = max_generations
        self.mutation_rate = mut_rate
        self.time_limit = time_limit
        self.first_aid = first_aid_time
        self.home = (0, 0)

        self.victim_ids = list(victims_cluster.keys())
        self.pathfinder = Dijkstra(self.home, self.map, cost_line, cost_diag)
        self.dist_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        nodes = [self.home] + [self.victims[vid][0] for vid in self.victim_ids]
        size = len(nodes)
        dist_mat = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    _, cost = self.pathfinder.calc_plan(nodes[i], nodes[j])
                    dist_mat[i][j] = cost * 1.1 if cost != -1 else float('inf')
        return dist_mat

    def _position_to_index(self, pos):
        if pos == self.home:
            return 0
        return [self.victims[vid][0] for vid in self.victim_ids].index(pos) + 1

    def _route_cost(self, route):
        total = 0
        current_idx = 0
        for vid in route:
            next_idx = self._position_to_index(self.victims[vid][0])
            total += self.dist_matrix[current_idx][next_idx]
            current_idx = next_idx
        total += self.dist_matrix[current_idx][0]
        return total

    def _evaluate_route(self, route):
        elapsed_time = 0
        severity_score = 0
        current_pos = self.home

        for vid in route:
            idx_curr = self._position_to_index(current_pos)
            idx_next = self._position_to_index(self.victims[vid][0])

            travel_time = self.dist_matrix[idx_curr][idx_next]
            treatment_time = self.first_aid
            return_time = self.dist_matrix[idx_next][0]

            if elapsed_time + travel_time + treatment_time + return_time > self.time_limit:
                continue

            elapsed_time += travel_time + treatment_time
            vitals = self.victims[vid][1]
            severity_score += vitals[6] * (1 + (4 - vitals[7]) / 4)
            current_pos = self.victims[vid][0]

        elapsed_time += self.dist_matrix[self._position_to_index(current_pos)][0]

        if elapsed_time > self.time_limit:
            over_time = elapsed_time - self.time_limit
            severity_score *= max(0, 1 - (over_time / self.time_limit) ** 2)

        return severity_score

    def _init_population(self):
        population = []

        # Heurística 1: Severidade / distância
        priority_list = sorted(
            self.victim_ids,
            key=lambda v: self.victims[v][1][6] / self.dist_matrix[0][self._position_to_index(self.victims[v][0])],
            reverse=True,
        )
        population.append(priority_list)

        # Heurística 2: Vizinho mais próximo modificado
        nn_seq = self._nearest_neighbor(self.victim_ids)
        population.append(nn_seq)

        # Heurística 3: Agrupamento geográfico
        geo_cluster = sorted(self.victim_ids, key=lambda v: sum(self.victims[v][0]))
        population.append(geo_cluster)

        # Preenche o restante com rotas aleatórias
        for _ in range(self.population_size - 3):
            population.append(random.sample(self.victim_ids, len(self.victim_ids)))

        return population

    def _nearest_neighbor(self, victims):
        remaining = victims.copy()
        seq = []
        current_pos = self.home

        while remaining:
            nearest_vic = None
            min_dist = float("inf")
            for v in remaining:
                dist = self.dist_matrix[self._position_to_index(current_pos)][self._position_to_index(self.victims[v][0])]
                if dist < min_dist:
                    min_dist = dist
                    nearest_vic = v
            if nearest_vic is None:
                break
            seq.append(nearest_vic)
            current_pos = self.victims[nearest_vic][0]
            remaining.remove(nearest_vic)

        return seq

    def _select_parents(self, candidates, scores):
        tour_size = max(2, int(len(candidates) * 0.1))
        parents = []
        for _ in range(2):
            group = random.sample(list(zip(candidates, scores)), tour_size)
            group.sort(key=lambda x: x[1], reverse=True)
            parents.append(group[0][0])
        return parents

    def _edge_recombination_crossover(self, mom, dad):
        edges = defaultdict(set)
        for seq in [mom, dad]:
            for i, v in enumerate(seq):
                prev_v = seq[i - 1] if i > 0 else None
                next_v = seq[i + 1] if i < len(seq) - 1 else None
                edges[v].update({prev_v, next_v})

        child = [mom[0]]
        current = child[0]

        while len(child) < len(mom):
            neighbors = edges[current]
            chosen = None
            for nb in neighbors:
                if nb not in child and nb is not None:
                    chosen = nb
                    break
            if chosen is None:
                remaining = [v for v in mom if v not in child]
                chosen = random.choice(remaining)
            child.append(chosen)
            current = chosen

        return child

    def _mutate(self, route):
        if random.random() < self.mutation_rate:
            if random.random() < 0.5 and len(route) > 2:
                i, j = sorted(random.sample(range(len(route)), 2))
                route[i:j + 1] = reversed(route[i:j + 1])
            else:
                critical_indices = [i for i, v in enumerate(route) if self.victims[v][1][7] == 1]
                if len(critical_indices) >= 2:
                    i, j = random.sample(critical_indices, 2)
                    route[i], route[j] = route[j], route[i]
        return route

    def run(self):
        population = self._init_population()
        best_sol = None
        best_score = -float("inf")

        for _ in range(self.max_gens):
            scores = [self._evaluate_route(r) for r in population]

            for idx, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_sol = population[idx]

            next_gen = [best_sol.copy()]

            while len(next_gen) < self.population_size:
                p1, p2 = self._select_parents(population, scores)
                child = self._edge_recombination_crossover(p1, p2)
                child = self._mutate(child)
                next_gen.append(child)

            population = next_gen

        return best_sol
