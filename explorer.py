# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import math

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc, dir):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        
        self.current_plan = []     # plano de movimento a ser executado   
        self.return_plan = []      # plano de retorno a base
        self.dir = dir             # diferencia as direções de cada explorador
        dir_map = {
            1: (0, 1),   # Norte
            2: (1, 0),   # Leste
            3: (0, -1),  # Sul
            4: (-1, 0),  # Oeste
        }
        # Define a direção inicial
        self.preferred_dir = dir_map.get(dir, (0, 1))  # padrão: Norte

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
    
    def find_unmapped_position(self):
        """
        Busca uma célula conhecida que tenha uma célula vizinha ainda não mapeada.
        Retorna a célula conhecida e a desconhecida que devem ser alcançadas.
        """
        menor_distancia = float('inf')
        melhor_caminho = None

        for (x, y), (_, _, walls) in self.map.data.items():
            for dir_index, (dx, dy) in Explorer.AC_INCR.items():
                nx, ny = x + dx, y + dy
                if walls[dir_index] == VS.CLEAR and (nx, ny) not in self.map.data:
                    distancia = math.sqrt((self.x - x)**2 + (self.y - y)**2)
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        melhor_caminho = (nx, ny), (x, y)

        return melhor_caminho if melhor_caminho else ((0, 0), (0, 0))
    
    def rotate_preferred_direction(self):
        dx, dy = self.preferred_dir
        self.preferred_dir = (dy, -dx)  # rotaciona 90° sentido horário
            
    def explore(self):
        """
        Esta função define o comportamento do explorador em tempo de execução.
        A cada ciclo, o agente decide para onde se mover com base em informações
        do ambiente parcial (mapa local) e sua direção preferida.
        
        Passo a passo:
        1. O agente analisa os vizinhos imediatos e verifica se existe alguma célula
           ainda não mapeada.
            -> Se houver, ele prioriza a que estiver mais alinhada com sua direção preferida
               (definida no início com base no índice do explorador).
            -> Ao encontrar, executa o movimento diretamente e atualiza o mapa.
            -> Caso ele encontre muitas paredes em sua volta, modifica sua direção preferida,
               evitando ficar preso em cantos
            
        2. Se todos os vizinhos conhecidos já foram explorados, ele procura uma célula
           do mapa conhecida que esteja adjacente a uma célula ainda desconhecida.
           → Esse é o alvo de exploração (fronteira).
           → O agente então utiliza A* para planejar um caminho até essa célula conhecida
             e adiciona um passo extra para atingir a célula desconhecida vizinha.
        
        3. Se já houver um plano de movimento (calculado anteriormente), ele apenas
           executa o próximo passo do plano.
        """

        walls = self.check_walls_and_lim()
        if walls.count(VS.OBST_WALL) >= 5:
            self.rotate_preferred_direction()

        # Define uma lista de vizinhos desconhecidos
        unknown_neighbors = []
        for dir_index, (dx, dy) in Explorer.AC_INCR.items():
            nx, ny = self.x + dx, self.y + dy
            if walls[dir_index] == VS.CLEAR and (nx, ny) not in self.map.data:
                unknown_neighbors.append((dx, dy))

        if unknown_neighbors:
            # Ordena por alinhamento com a direção preferida (dot product negativo = mais alinhado)
            px, py = self.preferred_dir
            unknown_neighbors.sort(key=lambda d: -(d[0]*px + d[1]*py))
            dx, dy = unknown_neighbors[0]
            self.execute_move(dx, dy)
            return

        # Se não houver vizinhos desconhecidos, busca a célula conhecida mais próxima de um desconhecido
        target, entry = self.find_unmapped_position()
        if entry:
            # Objetivo conhecido é a célula atual
            if (self.x, self.y) == entry:
                dx = target[0] - self.x
                dy = target[1] - self.y
                self.current_plan = [(dx, dy)]
            # Objetivo conhecido está em uma célula mais distante
            else:
                # Calcula o melhor caminho até o objetivo conhecido
                path, cost = astar_search((self.x, self.y), entry, self.map, self.COST_LINE, self.COST_DIAG)
                if path is not None and cost >= 0:
                    dx = target[0] - entry[0]
                    dy = target[1] - entry[1]
                    # Plano até o objetivo conhecido + objetivo a ser descoberto
                    self.current_plan = path + [(dx, dy)]

        # 3. Executa plano se houver
        if self.current_plan:
            dx, dy = self.current_plan.pop(0)
            self.execute_move(dx, dy)

    # Execução padrão de um movimento
    def execute_move(self, dx, dy):
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Adiciona a parede ao mapa se não conseguir se mover
        if result == VS.BUMPED:
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            return

        if result == VS.EXECUTED:
            self.x += dx
            self.y += dy

            # Checa por vítimas
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)

            difficulty = (rtime_bef - rtime_aft)
            difficulty /= self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG

            # Adiciona a célula ao mapa
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

    def come_back(self):
        print(f"{self.NAME} - Retornando para base!")
        
        # Cácula o plano de retorno se ainda não houver
        if not self.return_plan:
            path, cost = astar_search((self.x, self.y), (0, 0), self.map, self.COST_LINE, self.COST_DIAG)
            if path:
                self.return_plan = path
            else:
                print(f"{self.NAME}: Falha ao planejar retorno à base.")
                return

        # Executa o plano de retorno a base
        if self.return_plan:
            dx, dy = self.return_plan.pop(0)
            result = self.walk(dx, dy)

            # Se o plano falhar, recalcula
            if result == VS.BUMPED:
                print(f"{self.NAME}: bump inesperado ao retornar em ({self.x+dx}, {self.y+dy})")
                self.return_plan = []  # força replanejamento no próximo ciclo
                return

            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy
            
    def deliberate(self) -> bool:
        """ Define o comportamento do agente em cada ciclo """

        # Se já existe um plano de retorno, apenas executa ele
        if self.return_plan:
            self.come_back()
            return True

        # Margem de segurança para leitura e eventual replanejamento
        time_tolerance = 2 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # Calcula caminho e custo até a base
        path_to_base, cost_to_base = astar_search((self.x, self.y), (0, 0), self.map, self.COST_LINE, self.COST_DIAG)
        rtime = self.get_rtime()  # energia restante

        print(f"{self.NAME} - Bateria: {rtime:.2f} | Custo para voltar: {cost_to_base:.2f}")

        # Se ainda há energia suficiente para explorar com segurança
        if cost_to_base >= 0 and rtime > cost_to_base + time_tolerance:
            self.explore()
            return True

        # Se já está na base
        if (self.x, self.y) == (0, 0):
            self.resc.sync_explorers(self.map, self.victims)
            return False

        # Começa a retornar
        self.come_back()
        return True

from heapq import heappush, heappop

def astar_search(start, goal, map_instance, cost_line=1.0, cost_diag=1.5):
    """
    Algoritmo A* adaptado para trabalhar com mapa parcial (map_instance.data).

    Parâmetros:
    - start: tupla (x, y) posição inicial
    - goal: tupla (x, y) posição objetivo
    - map_instance: instância do mapa com método get_actions_results(pos) e get_difficulty(pos)

    Retorna:
    - path: lista de movimentos (dx, dy) até o destino
    - total_cost: custo estimado do caminho
    """
    def heuristic(a, b):
        # Distância Euclidiana
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    increments = {
        0: (0, -1),  1: (1, -1), 2: (1, 0),  3: (1, 1),
        4: (0, 1),   5: (-1, 1), 6: (-1, 0), 7: (-1, -1)
    }

    # Verifica se as posições existem no mapa parcial
    if goal not in map_instance.data or start not in map_instance.data:
        return [], -1

    open_set = []
    heappush(open_set, (heuristic(start, goal), 0, start, []))
    visited = set()

    while open_set:
        est_total_cost, cost_so_far, current, path = heappop(open_set)

        if current == goal:
            return path, cost_so_far

        if current in visited:
            continue
        visited.add(current)

        for dir_index, (dx, dy) in increments.items():
            neighbor = (current[0] + dx, current[1] + dy)

            # Só expande vizinhos conhecidos no mapa
            if neighbor not in map_instance.data:
                continue

            # Verifica se há parede bloqueando o movimento
            walls = map_instance.get_actions_results(current)
            if not walls or walls[dir_index] != 0:  # VS.CLEAR == 0
                continue

            difficulty = map_instance.get_difficulty(neighbor)
            # Define o custo real do passo baseado na direção
            step_cost = cost_line * difficulty if dx == 0 or dy == 0 else cost_diag * difficulty
            new_cost = cost_so_far + step_cost
            new_path = path + [(dx, dy)]
            est_total = new_cost + heuristic(neighbor, goal)

            heappush(open_set, (est_total, new_cost, neighbor, new_path))

    return [], -1  # Nenhum caminho encontrado
