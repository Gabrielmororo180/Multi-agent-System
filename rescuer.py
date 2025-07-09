##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import numpy as np
import csv
from map import Map
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from bfs import BFS
from genetic_seq import GASequencer
from collections import OrderedDict

from colorama import init, Fore
init()


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, model=None, use_model=False, clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.model = model
        self.use_model = use_model
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited 
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster 
        
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)


    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self, k=4, max_iter=100, tolerance=0.001):
        """
        Método otimizado de clusterização usando K-Means++ para inicialização de centróides.
        
        Parâmetros:
        - k: número de clusters (por padrão 4)
        - max_iter: número máximo de iterações para convergência (por padrão 100)
        - tolerance: limiar para a mudança dos centróides (por padrão 0.001)
        
        Retorna:
        - final_clusters: Lista de clusters com as vítimas atribuídas a eles.
        """
        
        # Etapa 1: Coletar posições
        positions = []
        for vic_id, values in self.victims.items():
            positions.append(values[0])

        print(f"{self.NAME}: Coletou {len(positions)} vítimas para clusterizar.")

        # Etapa 2: Inicialização dos centróides com K-Means++
        centroids = self._kmeans_initialization(positions, k)

        # Etapa 3: Algoritmo K-Means
        clusters = {}
        iter_count = 0

        for it in range(max_iter):
            # Etapa 3.1: Atribuir vítimas ao centróide mais próximo
            for i, vic_id in enumerate(self.victims.keys()):
                pos = positions[i]
                closest_centroid_id = self._assign_to_closest_centroid(pos, centroids)
                clusters[vic_id] = closest_centroid_id

            # Etapa 3.2: Recalcular centróides e verificar convergência
            new_centroids, centroid_movement = self._recalculate_centroids(k, centroids, clusters, positions)

            # Condição de parada: se o movimento for menor que a tolerância
            if centroid_movement < tolerance:
                break

            centroids = new_centroids
            iter_count = it + 1

        final_clusters = self._group_victims_by_cluster(k, clusters)
            
        if iter_count == max_iter -1:
            print(f"{self.NAME}: Atingiu o número máximo de iterações ({max_iter}).")

        ### ADDED: Log final cluster details
        for i, cluster in enumerate(final_clusters):
            victim_ids = list(cluster.keys())
            print(f"{Fore.GREEN}{self.NAME}: Cluster {i} contains {len(cluster)} victims: {victim_ids[:5]}{'...' if len(victim_ids) > 5 else ''}{Fore.RESET}")
        
        return final_clusters

    def _kmeans_initialization(self, positions, k):
        """
        Inicialização otimizada dos centróides usando o K-Means++.
        """
        centroids = []
        # Escolher o primeiro centróide aleatoriamente
        first_centroid_idx = random.randint(0, len(positions) - 1)
        centroids.append(positions[first_centroid_idx])

        for _ in range(1, k):
            dist_sq = []
            for pos in positions:
                # Calcula a distância quadrada mínima de um ponto a qualquer centróide existente
                min_dist_sq = min([self._euclidean_distance(pos, c)**2 for c in centroids])
                dist_sq.append(min_dist_sq)
            
            # Escolhe o próximo centróide com uma probabilidade proporcional à distância quadrada
            probabilities = np.array(dist_sq) / sum(dist_sq)
            cumulative_probabilities = np.cumsum(probabilities)
            r = random.random()
            
            chosen_idx = -1
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    chosen_idx = i
                    break
            
            # Garante que um índice seja escolhido
            if chosen_idx == -1:
                chosen_idx = len(positions) -1
            
            centroids.append(positions[chosen_idx])

        return centroids

    def _assign_to_closest_centroid(self, pos, centroids):
        """
        Atribui uma vítima ao centróide mais próximo com base na distância.
        """
        min_dist = float('inf')
        closest_centroid_id = -1
        for i, centroid in enumerate(centroids):
            dist = self._euclidean_distance(pos, centroid)
            if dist < min_dist:
                min_dist = dist
                closest_centroid_id = i
        return closest_centroid_id

    def _recalculate_centroids(self, k, old_centroids, clusters, positions):
        """
        Recalcula os centróides 2D com base na média das posições das vítimas atribuídas.
        """
        new_centroids = []
        # Mapeia vic_id para sua posição para facilitar a busca
        vic_positions = {vic_id: pos for vic_id, pos in zip(self.victims.keys(), positions)}
        
        for i in range(k):
            # Obtém os IDs das vítimas neste cluster
            cluster_victims_ids = [vic_id for vic_id, cluster_id in clusters.items() if cluster_id == i]
            
            if not cluster_victims_ids:
                # Se o cluster estiver vazio, mantém o centróide antigo
                new_centroids.append(old_centroids[i])
                continue

            # Calcula a média das coordenadas x e y
            sum_x, sum_y = 0, 0
            for vic_id in cluster_victims_ids:
                pos = vic_positions[vic_id]
                sum_x += pos[0]
                sum_y += pos[1]
            
            count = len(cluster_victims_ids)
            new_centroid = (sum_x / count, sum_y / count)
            new_centroids.append(new_centroid)

        # Calcula o movimento total dos centróides para verificar a convergência
        total_movement = sum(self._euclidean_distance(old, new) for old, new in zip(old_centroids, new_centroids))
        
        return new_centroids, total_movement

    def _group_victims_by_cluster(self, k, clusters):
        """
        Agrupa as vítimas por seus respectivos clusters.
        """
        final_clusters = [{} for _ in range(k)]
        for vic_id, cluster_id in clusters.items():
            if cluster_id != -1: # Garante que a vítima foi atribuída a um cluster válido
                final_clusters[cluster_id][vic_id] = self.victims[vic_id]
        return final_clusters

    def _euclidean_distance(self, point1, point2):
        """
        Calcula a distância euclidiana entre dois pontos.
        """
        return np.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))

    def predict_severity_and_class(self):
        """ @TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
            This method should add the vital signals(vs) of the self.victims dictionary with these two values.

            This implementation assigns random values to both, severity value and class"""

        with open('file_predict.txt', 'w') as f_predict:
            for vic_id, values in self.victims.items():
                # Step 1: Extract features (first 5 vital signals: sp, dp, qp, pf, rf)
                features = values[1][1:6]  # Assumes vital signals are ordered as in PredictionModel

                # Step 2: Scale features using the model's scaler
                features_scaled = self.model.scaler.transform([features])

                # Step 3: Predict severity value using the regressor
                severity_value = self.model.regressor.predict(features_scaled)[0]

                # Step 4: Predict severity class using the classifier
                severity_class = self.model.classifier.predict(features_scaled)[0]

                # Step 5: Write predictions to file for test set victims (if test_ids available)
                x, y = values[0]  # Victim's coordinates
                f_predict.write(f"{vic_id},{x},{y},{severity_value:.2f},{severity_class}\n")

                # Step 6: Append predictions to the victim's vital signals
                if self.use_model:
                    values[1].extend([severity_value, severity_class])

                else:
                    for vic_id, values in self.victims.items():
                        severity_value = random.uniform(0.1, 99.9)          # to be replaced by a regressor  
                        severity_class = random.randint(1, 4)               # to be replaced by a classifier
                        values[1].extend([severity_value, severity_class])  # append to the list of vital signals; values is a pair( (x,y), [<vital signals list>] )


    def sequencing(self):
        new_sequences = []
        for cluster in self.sequences:
            if not cluster:
                continue
            # Run GA to find the best sequence
            sequencer = GASequencer(
                cluster, 
                self.map, 
                self.COST_LINE, 
                self.COST_DIAG,
                  pop_size=200,          # <== aqui, não population_size
                max_generations=500,
                mut_rate=0.3,
                time_limit=self.TLIM,
                first_aid_time=self.COST_FIRST_AID
            )
            best_sequence = sequencer.run()
            # Convert sequence to OrderedDict
            ordered_cluster = OrderedDict()
            for vic_id in best_sequence:
                ordered_cluster[vic_id] = cluster[vic_id]
            new_sequences.append(ordered_cluster)

            ### ADDED: Log sequence result
            print(f"{Fore.GREEN}{self.NAME}: Sequence generated: {[vic_id for vic_id in best_sequence]}{Fore.RESET}")

        self.sequences = new_sequences


    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""


        # let's instantiate the breadth-first search
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        start = (0,0) # always from starting at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal

        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        self.plan = self.plan + plan
        self.plan_rtime = self.plan_rtime - time
           

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        ### ADDED: Log map and victim update
        print(f"{Fore.GREEN}{self.NAME}: Received map {self.received_maps}/{self.nb_of_explorers}, cells={len(explorer_map.data)}, victims={len(victims)}{Fore.RESET}")
        print(f"{self.NAME}: Merged map, total cells={len(self.map.data)}")

        if self.received_maps == self.nb_of_explorers:
            ### ADDED: Log completion of map collection
            print(f"{Fore.RED}{self.NAME}: All maps received, total victims={len(self.victims)}, total map cells={len(self.map.data)}{Fore.RESET}")

            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1
  
            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of 
            self.clusters = [clusters_of_vic[0]]  # the first one

            ### ADDED: Log master rescuer cluster assignment
            print(f"{Fore.RED}{self.NAME}: Assigned cluster 1 with {len(self.clusters[0])} victims{Fore.RESET}")

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):    
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, self.model, self.use_model, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map     # each rescuer have the map

            
            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters         

            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()         # the sequencing will reorder the cluster
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1 
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

                    ### ADDED: Log sequence saving
                    print(f"{self.NAME}: Saved sequence {i+1+j*10} for Rescuer_{i+1}")
            
                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method 
         
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True
