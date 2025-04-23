from envs.game_elements import *

import gym
from gym import spaces

import matplotlib.pyplot as plt
import numpy as np

# Création de l'environnement
class TankEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_x = 20, max_y = 20, max_enemies_on_screen = 5, total_ennemies_to_kill = 20):
        super(TankEnv, self).__init__()
        
        self.max_x = max_x # Largeur de la grille
        self.max_y = max_y # Hauteur de la grille
        self.max_enemies_on_screen = max_enemies_on_screen
        self.max_projectiles = max_x * max_y

        self.total_ennemies_to_kill = total_ennemies_to_kill
        self.initial_ennemies = 2

        # assertions
        assert max_x > 0
        assert max_y > 0
        assert max_enemies_on_screen > 0
        assert max_enemies_on_screen <= total_ennemies_to_kill
        assert (max_enemies_on_screen + 1) * 9 * 2 <= max_x * max_y 

        # Define action space
        self.action_space = spaces.Discrete(6)  # 0: up, 1: right, 2: down, 3: left, 4: stay, 5: shoot
        
        # Define observation space
        dtypes = np.dtype('int32')
        self.observation_space = spaces.Dict({
            'player': spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.max_x, self.max_y, 4]), dtype=dtypes),  # x, y, direction
            'enemies': spaces.Box(low=np.zeros((self.max_enemies_on_screen, 3), dtype=dtypes), high=np.array([self.max_x, self.max_y, 4] * self.max_enemies_on_screen).reshape(self.max_enemies_on_screen, 3), dtype=dtypes),
            'projectiles': spaces.Box(low=np.zeros((self.max_projectiles, 4), dtype=dtypes), high=np.array([self.max_x, self.max_y, 4, 1] * self.max_projectiles).reshape(self.max_projectiles, 4), dtype=dtypes)  # x, y, direction, from (0: player, 1: enemy)
        })

        # Define state
        self.state = {
            'player': set(),
            'enemies': set(), # (Tank(0, 0, np.array([0, 0, 1, 0])), Tank(0, 0, np.array([0, 0, 1, 0])), ...)
            'projectiles': set() # (Projectile(0, 0, np.array([0, 0, 1, 0]), label=0), Projectile(0, 0, np.array([0, 0, 1, 0]), label=1), ...)
        }
        

        # Set of all positions occupied by tanks
        self.occupied_positions = set()

        self.probability_new_enemy = 0.01

        #rewards value
        self.reward_enemy_killed = 1
        self.reward_player_dead = -1
        self.reward_wasted_projectile = -0.05
        self.reward_target_on_sight=0.5
        self.reward_nothing = -0.01
        self.timestep = -0.01

        self.done = False
        self.info = {}
        
    def reset(self):
        self.occupied_positions = set()
        self.done = False

        # placer le joueur
        ## strat: de manière aléatoire
        x = np.random.randint(0, self.max_x)
        y = np.random.randint(0, self.max_y)

        direction = np.zeros(4, dtype=int)
        direction[np.random.randint(0, 4)] = 1 # une direction aléatoire
        initial_kill_streak=0
        player = Tank(x, y, direction,initial_kill_streak, label=0)
        self.state['player'] = player

        self.occupied_positions.add((x, y))

        # placer les ennemis, attention aux collisions
        ## strat: self.initial_ennemies ennemis de manière aléatoire
        ennemies = set()
        for i in range(self.initial_ennemies):
            placed = False
            while not placed:
                x = np.random.randint(0, self.max_x)
                y = np.random.randint(0, self.max_y)

                boxes = [(x + i, y + j) for i in range(-2, 3) for j in range(-2, 3)]
                if any(box in self.occupied_positions for box in boxes):
                    continue
                
                direction = np.zeros(4, dtype=int)
                direction[np.random.randint(0, 4)] = 1

                ennemies.add(Tank(x, y, direction,kill_streak=0, label=1))
                self.occupied_positions.add((x, y))
                placed = True

        self.state['enemies'] = ennemies

        print("#### environnement reset successfully ####")

    

    def step(self, action):
        #Tells if a target is on sight or not 
        def is_target_in_sight(player, enemies, max_range=22):
            px, py, direction, _ = player.info()

            # Liste de toutes les cases occupées par les ennemis (hitbox 5x5)
            enemy_boxes = set()
            for enemy in enemies:
                ex, ey, _, _ = enemy.info()
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        enemy_boxes.add((ex + dx, ey + dy))

            # Simuler le chemin du tir dans la direction actuelle
            for i in range(1, max_range):
                if direction == 0:  # ↑ haut
                    tx, ty = px, py - i
                elif direction == 1:  # → droite
                    tx, ty = px + i, py
                elif direction == 2:  # ↓ bas
                    tx, ty = px, py + i
                elif direction == 3:  # ← gauche
                    tx, ty = px - i, py

                if (tx, ty) in enemy_boxes:
                    return True  # Un ennemi est sur la trajectoire

            return False
        
        reward = self.timestep

        ## annulation des projectiles qui se touchent si necessaire
        projectiles = list(self.state['projectiles'])
        for i in range(len(projectiles)):
            for j in range(i+1, len(projectiles)):
                if projectiles[i].x == projectiles[j].x and projectiles[i].y == projectiles[j].y and projectiles[i].label != projectiles[j].label:
                    # same position, different players
                    try:
                        self.state['projectiles'].remove(projectiles[i])
                        self.state['projectiles'].remove(projectiles[j])
                    except:
                        pass

        # Rajouter 1 ennemi si le nombre d'ennemis actifs est inférieur à max_enemies
        ## strat: de manière aléatoire avec une probabilité de self.probability_new_enemy
        if (len(self.state['enemies']) < self.max_enemies_on_screen and np.random.rand() < self.probability_new_enemy) or len(self.state['enemies']) == 0:
            placed = False
            while not placed:
                x = np.random.randint(0, self.max_x)
                y = np.random.randint(0, self.max_y)

                boxes = [(x + i, y + j) for i in range(-2, 3) for j in range(-2, 3)]
                if any(box in self.occupied_positions for box in boxes):
                    continue
                
                direction = np.zeros(4, dtype=int)
                direction[np.random.randint(0, 4)] = 1

                self.state['enemies'].add(Tank(x, y, direction,kill_streak=0, label=1))
                self.occupied_positions.add((x, y))
                placed = True

        # Nettoyer les ennemis tombés, les projectiles utilisés
        ## reward_ennemy_killed pour chaque ennemi tombé
        for enemy in list(self.state['enemies']):
            boxes = [(enemy.x + i, enemy.y + j) for i in range(-2, 3) for j in range(-2, 3)]
            for projectile in list(self.state['projectiles']):
                if (projectile.x, projectile.y) in boxes and projectile.label == 0:
                    self.state['enemies'].remove(enemy)
                    self.occupied_positions.remove((enemy.x, enemy.y))
                    self.state['projectiles'].remove(projectile)
                    reward += self.reward_enemy_killed
                    self.state['player'].kill_streak+=1
                    break

        # Verifier si le joueur est mort
        ## le joueur est mort: done = True, reward = reward_player_dead
        ## le joueur n'est pas mort: done = False
        boxes = [(self.state['player'].x + i, self.state['player'].y + j) for i in range(-2, 3) for j in range(-2, 3)]
        ennemies_projectiles_positions = [(projectile.x, projectile.y) for projectile in self.state['projectiles'] if projectile.label == 1]
        if any(pos in boxes for pos in ennemies_projectiles_positions):
            self.done = True
            reward += self.reward_player_dead

        ##################### update #####################
            
        # # Mettre à jour l'état du joueur
        ## strat: en fonction de l'action
        ## reward si l'action est "stay" ou "shoot", 0 sinon
        bondaries = {
            'max_x': self.max_x,
            'max_y': self.max_y,
        }
        self.state['player'].update(action, self.state, self.occupied_positions, bondaries)
        if action == 4:
            reward += self.reward_nothing

        # Si l'agent tire sur une cible dans son viseur alors il est récompensé et pénalisé si il tire sans réfléchir
        if action == 5:
                if is_target_in_sight(self.state['player'], self.state['enemies']):
                    reward += self.reward_target_on_sight
                else:
                    reward+= self.reward_wasted_projectile

        # Mettre à jour l'état des ennemis
        ## strat: de maniere aleatoire
        for enemy in self.state['enemies']:
            enemy.update_strategic(self.state, self.occupied_positions, bondaries, strategy=2)

        # Mettre à jour l'état des projectiles
        ## position
        for projectile in list(self.state['projectiles']): # list() pour éviter les modifications en cours de parcours
            projectile.update(self.state, bondaries)
        
        ##################### update done #####################
            
        return self.state, reward, self.state['player'].kill_streak, self.done, {}

    def render(self, mode='human'):
        # Créer une matrice M de taille max_x * max_y + un padding de 1 de chaque côté
        rows = self.max_y + 2
        cols = self.max_x + 2
        M = np.ones((rows, cols, 3), dtype=np.uint8) * 240 # white background
        # remplir la matrice avec les éléments de l'environnement
        def fill_tank(tank, color):
            # print(f"#### filling tank {tank.coord_and_dir} with color {color} ####")
            x, y, dir, _ = tank.info()
            x += 1 # because of the padding
            y += 1 # because of the padding

            M[y, x] = color 
            if dir in [1, 2]:
                M[y-1, x-1, :] = color
            if dir in [0, 1, 3]:
                M[y-1, x, :] = color
            if  dir in [2, 3]:
                M[y-1, x+1, :] = color
            if dir in [0, 2, 3]:
                M[y, x-1, :] = color
            if dir in [0, 1, 2]:
                M[y, x+1, :] = color
            if dir in [0, 1]:
                M[y+1, x-1, :] = color
            if dir in [1, 2, 3]:
                M[y+1, x, :] = color
            if dir in [0, 3]:
                M[y+1, x+1, :] = color
        ## remplir avec le joueur
        fill_tank(self.state['player'], [92, 184, 92])
                
        ## remplir avec les ennemis
        for enemy in self.state['enemies']:
            fill_tank(enemy, [240, 173, 78])

        ## remplir avec les projectiles
        for projectile in self.state['projectiles']:
            x, y, dir, label = projectile.info()
            if label == 0:
                M[y+1, x+1, :] = [217, 100, 79]
            else:
                M[y+1, x+1, :] = [217, 83, 79]

        # return frame
        return M
    
    def plot_render(self):
        M = self.render()
        plt.axis('off')
        plt.imshow(M)
        plt.show()