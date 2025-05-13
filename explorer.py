# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from bfs import BFS

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
        print(f"Explorador {dir}")
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        
        self.disperse_steps = 10   # quantos passos dar antes de explorar
        self.current_plan = []     # plano de movimento a ser executado   
        self.bfs = BFS(self.map)   # bfs para calcular o plano a ser executado 
        self.dir = dir             # diferencia as direções de cada explorador
        dir_map = {
            1: (0, -1),   # Norte
            2: (1, 0),    # Leste
            3: (0, 1),    # Sul
            4: (-1, 0),   # Oeste
        }
        # Define a direção inicial
        self.preferred_dir = dir_map.get(dir, (0, -1))  # padrão: Norte

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    
    def find_unmapped_position(self):
        """
        Busca uma célula conhecida que tenha uma célula vizinha ainda não mapeada.
        Retorna a célula conhecida que deve ser alcançada para descobrir o novo vizinho.
        """
        for (x, y), (_, _, walls) in self.map.data.items():
            for dir_index, (dx, dy) in Explorer.AC_INCR.items():
                nx, ny = x + dx, y + dy
                if walls[dir_index] == VS.CLEAR and (nx, ny) not in self.map.data:
                    return (nx, ny), (x, y)
                
        return (0, 0), (0, 0)
            
    def explore(self):
        walls = self.check_walls_and_lim()

        # dispersa os exploradores para cada direção
        if self.disperse_steps > 0:
            dx, dy = self.preferred_dir
            self.execute_move(dx, dy)
            self.disperse_steps -= 1
            return

        # Tenta ir diretamente para uma célula vizinha desconhecida e acessível
        for dir_index, (dx, dy) in Explorer.AC_INCR.items():
            nx, ny = self.x + dx, self.y + dy
            if walls[dir_index] == VS.CLEAR and (nx, ny) not in self.map.data:
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
                path, cost = self.bfs.search((self.x, self.y), entry)
                if path is not None and cost >= 0:
                    dx = target[0] - entry[0]
                    dy = target[1] - entry[1]
                    # Plano até o objetivo conhecido + objetivo a ser descoberto
                    self.current_plan = path + [(dx, dy)]

        # 3. Executa plano se houver
        if self.current_plan:
            dx, dy = self.current_plan.pop(0)
            self.execute_move(dx, dy)

    def execute_move(self, dx, dy):
        print(f"{dx} {dy}")
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.BUMPED:
            return

        if result == VS.EXECUTED:
            self.walk_stack.push((dx, dy))
            self.x += dx
            self.y += dy
            self.walk_time += (rtime_bef - rtime_aft)

            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)

            difficulty = (rtime_bef - rtime_aft)
            difficulty /= self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG

            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if  self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        print("Retornando a base")
        self.come_back()
        return True

