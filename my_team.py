# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import json

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

IS_TRAINING = False
NUM_EPISODES = 1000

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQLearner', second='DefensiveQLearner', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # print("hellouda")
    num_training = NUM_EPISODES
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ApproxQLearningAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        print("epsilon: 1.")
        super().__init__(index, time_for_computing)
        self.weights = util.Counter()

    def __del__(self) : # for training purposes only
        if IS_TRAINING:
            with open("weights_offensive.json", "w") as f:
                json.dump(self.weights, f)
            f.close()
            print("WEIGHTS SAVED TO FILE")

            with open("weights_log.json", "a+") as f_log :
                json.dump(self.weights, f_log)
            f_log.close()

    def register_initial_state(self, game_state):
        self.epsilon = 0.8      # prob of explore
        self.sigma = 0.999     # decay for epsilon-greedy
        self.alpha = 0.01       # learning rate
        self.gamma = 0.9        # discount
        self.num_episodes = NUM_EPISODES
        self.current_iteration = 0
        self.features = util.Counter()
        self.start = game_state.get_agent_position(self.index)

        if IS_TRAINING:
            with open("train_count.json", "r") as f_in :
                #print("R:", f_in.readline())
                count = json.load(f_in)
                count['num_training'] = int(count['num_training']) +1
                '''
                except :
                    print("MASSIVE ERROR")
                    count = {'num_training' : 1}
                '''
            f_in.close()

            with open("train_count.json", "w") as f_out :
                json.dump(count, f_out)
            f_out.close()

        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        action = 'Stop'

        if IS_TRAINING:         
        
            if util.flip_coin(self.epsilon): # EXPLORE
                # print("EXPLORE")
                # while action == 'Stop':
                action = random.choice(actions) 
            else:   # EXPLOIT
                # print("EXPLOIT")
                action = self.action_from_qvals(game_state) # take the action given by the policy

            next_game_state = self.get_successor(game_state, action)
            reward = self.get_reward(game_state, next_game_state)
            self.update_weights(game_state, action, reward, next_game_state)

            self.epsilon = max(0.05, self.epsilon*self.sigma)
            return action
        
        else:
            # gets action directly from the policy
            action = self.action_from_qvals(game_state)
            return action

    # update weights based on transition
    def update_weights(self, game_state, action, reward, next_game_state):
        # w += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a)) * phi(s,a)

        current_qval = self.get_qval(game_state, action)
        features = self.get_features(game_state, action)

        # max qval for next state
        actions = next_game_state.get_legal_actions(self.index)
        max_next_qval = max([self.get_qval(next_game_state, a) for a in actions]) if actions else 0

        # print("Updating weights... GOT REWARD : ", reward)
        # print(self.weights)

        td_err = reward + self.gamma * max_next_qval - current_qval
        td_err = max(-5.0, min(5.0, td_err))
        for f in features: self.weights[f] += self.alpha * td_err * features[f]

    # implementat en cada agent 
    def get_reward(self, current_game_state, next_game_state):
        util.raise_not_defined()

    def get_qval(self, game_state, action):
        #for item in self.weights.items() : print("W:", item)
        #for item in self.get_features(game_state, action).items() : print("F:", item)
        qval = self.get_features(game_state, action) * self.weights
        # print("qval: ", qval)
        return qval
 
    def action_from_qvals(self, game_state): # policy
        actions = game_state.get_legal_actions(self.index)
        if not actions: return None

        best_val = -util.sys.maxsize
        q_vals = {}

        for action in actions:
            val = self.get_qval(game_state, action)
            q_vals[action] = val
            if val > best_val: best_val = val
        # print("best val: ", best_val)
        best_actions = [a for a, v in q_vals.items() if v == best_val]
        
        return random.choice(best_actions)
        
    def val_from_qvals(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions: return 0.0

        best_value = -util.sys.maxsize
        for action in actions:
            value = self.get_qval(game_state, action)
            if value > best_value: best_value = value
        return best_value

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
        # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def get_features(game_state, action):
        util.raise_not_defined()

    # Calculates distance to the border between territories
    def get_distance_to_border(self, game_state, position):

        walls = game_state.get_walls()
        width = walls.width
        border_x = width // 2 
        
        min_dist = util.sys.maxsize
        
        for y in range(walls.height):
            if not walls[border_x][y]:
                dist = self.get_maze_distance(position, (border_x, y))
                if dist < min_dist:
                    min_dist = dist
        for y in range(walls.height):
            if not walls[border_x - 1][y]:
                dist = self.get_maze_distance(position, (border_x - 1, y))
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist if min_dist != util.sys.maxsize else 0
    

class OffensiveQLearner(ApproxQLearningAgent) :

    def __init__(self, index, time_for_computing=.1) :
        super().__init__(index, time_for_computing)
        if IS_TRAINING:         
            try:
                with open("weights_offensive.json", "r") as f:
                    self.weights = json.load(f)
                print("WEIGHTS SET TO PREVIOUS VALUES")
                f.close()
            except:
                print("WEIGHTS NOT FOUND. SET TO DEFAULT")
                self.initialize_weights_random()
            # print(self.weights, type(self.weights))
        else:
            # HARDCODED WEIGHTS 
            # self.weights = {"score_gain": 0.10034008345970119, "bias": 0.03444889065865667, "moved": 0.1, "is_action_stop": -0.25, "im_pacman": 0.027842228369004997, "edible_opp_1_far": 0.12392959553668438, "edible_opp_2_far": 0.08258998421190546, "dangerous_opp_1_far": -0.054180684590481365, "dangerous_opp_2_far": -0.03819569412378069, "min_edible_dist": 0.055372091949140836, "min_dangerous_dist": -0.09971103396311339, "min_capsule_dist": 0.5791876278431399, "min_food_dist": 0.8138611372890516, "min_dist_home": 0.06190230303216981, "3_from_border": 0.0032501879960864443, "carrying_food": 0.6927489266159432, "eating_food": 0.9}
            self.initialize_weights()

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)


    def initialize_weights(self):
        self.weights = util.Counter(
                    {"score_gain": 0.10319152349506475, 
                    "bias": 0.4244902760180313, 
                    "moved": 0.21548497357815952, 
                    "is_action_stop": -0.39345419309853, 
                    "im_pacman": 0.26881437940999553, 
                    "edible_opp_1_far": 0.06232410166522531, 
                    "edible_opp_2_far": 0.05430966534040033, 
                    "dangerous_opp_1_far": -0.15076050932597793, 
                    "dangerous_opp_2_far": -0.09874920054661913, 
                    "min_edible_dist": -0.21854789120883472, 
                    "min_dangerous_dist": 0.07564837431401258, 
                    "min_capsule_dist": -0.6893805666527831, 
                    "min_food_dist": -1.0177292144438388, 
                    "min_dist_home": -0.0854568663631837, 
                    "3_from_border": 0.09132695470425407, 
                    "carrying_food": 0.7636070061928109, 
                    "eating_food": 0.9}

                    )

    def initialize_weights_random(self):
        self.weights = util.Counter({
                    'score_gain' : 0.1,
                    'bias' : 0.035,
                    'moved': 0.1,
                    'is_action_stop': -0.25,
                    'im_pacman': 0.015,
                    'edible_opp_1_far': 0.08,
                    'edible_opp_2_far': 0.06,
                    'dangerous_opp_1_far': -0.1, 
                    'dangerous_opp_2_far': -0.06,
                    'min_edible_dist': -0.75,
                    'min_dangerous_dist': 0.1,
                    'min_capsule_dist': -0.09,
                    'min_food_dist': -0.12,
                    'min_dist_home': 0.75,
                    '3_from_border': 0.06,
                    'carrying_food' : 0.05,
                    'eating_food' : 0.1
                    })

        for feature in self.weights:
            self.weights[feature] *= random.uniform(0.9, 1.1)

    def get_features(self, game_state, action) :
        features = util.Counter()

        ag_state = game_state.get_agent_state(self.index)
        ag_pos = ag_state.get_position()

        next_game_state = self.get_successor(game_state, action)
        next_ag_state = next_game_state.get_agent_state(self.index)
        next_ag_pos = next_ag_state.get_position()
        next_ag_carrying = next_ag_state.num_carrying
        im_pacman = next_ag_state.is_pacman
        im_scared = next_ag_state.scared_timer

        features['score_gain'] = self.get_score(next_game_state) - self.get_score(game_state)
        features['score_gain'] /= 20
        features['bias'] = 1.
        features['im_pacman'] = next_ag_state.is_pacman
        features['is_action_stop'] = 1.0 if action == 'Stop' else 0.0
        features['moved'] = 0.0 if ag_pos == next_ag_pos else 1.0


        ##### Opponents ##### 
        opponents = self.get_opponents(game_state)

        # edibles:  if im pacman:   scared ghosts
        #           else:           pacman (except scared)
        # danger:   if im pacman:   ghosts (except scared)
        #           else:           none   (except scared)
        # CASES TO HANDLE:
        #   - im pacman | not
        #   - opp pacman | not
        #   - im scared | not
        #   - opp scared | not


        min_edible_dist = 9999
        min_dangerous_dist = 9999
        for opp in opponents:
            opp_state = next_game_state.get_agent_state(opp)
            opp_is_pacman = opp_state.is_pacman
            opp_is_scared = opp_state.scared_timer
            opp_pos = opp_state.get_position() # if opp is close 
            dist = self.get_maze_distance(opp_pos, next_ag_pos) if opp_pos != None else next_game_state.get_agent_distances()[opp]

            # CASE 1: Edibles
            if im_pacman and opp_is_scared:
                if min_edible_dist > dist : min_edible_dist = dist
                if dist < 2: features['edible_opp_1_far'] += 1
                elif dist == 2: features['edible_opp_2_far'] += 1
            elif not im_pacman and not im_scared:
                if min_edible_dist > dist : min_edible_dist = dist
                if dist < 2: features['edible_opp_1_far'] += 1
                elif dist == 2: features['edible_opp_2_far'] += 1
             
            # TU POTS!!!

            # CASE 2: Dangerous
            if im_pacman and not opp_is_scared:
                if min_dangerous_dist > dist : min_dangerous_dist = dist
                if dist < 2: features['dangerous_opp_1_far'] += 1
                elif dist == 2: features['dangerous_opp_2_far'] += 1
            elif not im_pacman and im_scared:
                if min_dangerous_dist > dist : min_dangerous_dist = dist
                if dist < 2: features['dangerous_opp_1_far'] += 1
                elif dist == 2: features['dangerous_opp_2_far'] += 1

            else: # if both pacman or both ghost
                if min_dangerous_dist > dist : min_dangerous_dist = dist
                if dist < 2: features['dangerous_opp_1_far'] += 1
                elif dist == 2: features['dangerous_opp_2_far'] += 1

        features['edible_opp_1_far'] /= 2
        features['edible_opp_2_far'] /= 2

        features['dangerous_opp_1_far'] /=  2
        features['dangerous_opp_2_far'] /=  2

        # f(x)= 1/sqrt(x) + 1/x^(0.15) - 1
        # fórmula pel valor de la distància: com més a prop, més important és. a partir de 11, la distància no es té en compte

        # features['min_edible_dist'] = max(((1 / (min_edible_dist**0.5)) + (1 / (min_edible_dist**0.15)) - 1), 0.0) if min_edible_dist != 0. else 0.0                # tindrà weight positiu
        # features['min_dangerous_dist'] = max(((1 / (min_dangerous_dist**0.5)) + (1 / (min_dangerous_dist**0.15)) - 1), 0.0) if min_dangerous_dist != 0. else 0.0    # tindrà weight negatiu

        features['min_edible_dist'] = 1.0 / (1.0 + min_edible_dist) if min_edible_dist < 20 else 0.0                        # tindrà weight positiu
        features['min_dangerous_dist'] = 1.0 / (1.0 + min_dangerous_dist) if min_dangerous_dist < 20 else 0.0               # tindrà weight negatiu

        capsule_list = self.get_capsules(next_game_state)
        min_capsule_dist = min([self.get_maze_distance(next_ag_pos, c) for c in capsule_list]) if capsule_list else 9999
        features['min_capsule_dist'] = 1.0 / (1.0 + min_capsule_dist) if min_capsule_dist < 20 else 0.0        # tindrà weight positiu
        
        # el food sempre és important, per tant la distància es considera sempre
        food_list = self.get_food(next_game_state).as_list()
        min_food_dist = min([self.get_maze_distance(next_ag_pos, food) for food in food_list]) if food_list else 9999
        features['min_food_dist'] = 1.0 / (1.0 + min_food_dist)

        # get back, get back to where you once belonged....
        dist_to_own = self.get_distance_to_border(next_game_state, next_ag_pos) +1
        # print("dist_home:", dist_to_own)
        # closer to border => higher value
        features['min_dist_home'] = 1. / (1.0 + dist_to_own) if im_pacman else 0.     # tindrà weight positiu             
        features['3_from_border'] = 1.0 if dist_to_own <= 3 else 0.0

        # features['min_dist_home'] = 1. / (1. + features['min_dist_home'] / 40)

        # quan porti 0, serà 0, a mida que augmenti, puja. 
        # h(x)=2-((1)/(sqrt(x+1)))-((1)/(x^(0.08)))
        features['carrying_food'] = max((2 - (1/ (next_ag_carrying + 1)**0.5) - (1/next_ag_carrying**0.08)), 0.0) if next_ag_carrying > 0. else 0.   # weight positiu

        return features
        # Get back Jojo!
    
    '''
    def get_reward(self, current_game_state, next_game_state):
    
        # REWARD LIST
        # -0.001      Step Penalty
        # -0.05       Still Penalty
        # +0.05       Being on Opponent side
        # +0.05       (*amount) Increasing Score
        # +0.5        Eating Food
        # +0.1        Get closer to Food
        # -1.0        Dying (getting to start position)
        # +0.5        Eating Opponent
        # -1.0        Being Eaten by Opponent


        reward = -0.001 # to prevent uselessness


        ag_state = current_game_state.get_agent_state(self.index)
        ag_pos = ag_state.get_position()
        ag_pacman = ag_state.is_pacman
        ag_carry = ag_state.num_carrying
        next_ag_state = next_game_state.get_agent_state(self.index)
        next_ag_pos = next_ag_state.get_position()
        next_ag_scared = next_ag_state.scared_timer
        next_ag_pacman = next_ag_state.is_pacman
        next_ag_carry = next_ag_state.num_carrying

        # penalty for still
        if ag_pos == next_ag_pos:
            reward -= 0.05
        if ag_pacman: reward +=0.05

        #### SCORE GAIN ####
        old_score = self.get_score(current_game_state)
        new_score = self.get_score(next_game_state)
        if new_score > old_score : reward += (new_score - old_score)*0.05


        #### EATING FOOD ####
        current_food = self.get_food(current_game_state).as_list()
        next_food = self.get_food(next_game_state).as_list()

        current_min_dist = min([self.get_maze_distance(ag_pos, food) for food in current_food]) if current_food else 40
        next_min_dist = min([self.get_maze_distance(next_ag_pos, food) for food in next_food]) if next_food else 40

        if len(current_food) > len(next_food) : reward += 0.1
        elif current_min_dist > next_min_dist : reward += 0.05



        #### DEATH ####
        # Check if we died (pacman -> ghost at start position)
        if next_ag_pos == self.start:
            reward -= 1.0 


        #### EATING OR BEING EATEN. OPPONENTS ####
        opponents = self.get_opponents(next_game_state)
        for opp in opponents:
            opp_state = next_game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()
            
            if opp_pos is None or opp_pos != next_ag_pos:
                continue
                
            # We're at the same position as opponent
            if opp_state.scared_timer > 0 :
                reward += 0.5
            else:
                reward -= 1.0 


        #### TERRITORY & SAFETY REWARDS ####
        # 'im_pacman' feature: penalty for being in enemy territory
        if next_ag_pacman:
            # reward -= 0.005  #  per step in enemy territory (risk)
            
            # 'min_dist_home' feature: bigger penalty if far from home
            dist_home = self.get_maze_distance(next_ag_pos, self.start)
            if dist_home > 10:
                reward -= 0.005 *(dist_home-10)
            
            if next_ag_carry > 0:
                dist = self.get_distance_to_border(next_game_state, next_ag_pos) 
            
                if dist <= 3:
                    reward += 0.05  # +0.05 for being near border with food

        

        # Bonus for returning to own side with food
        if ag_pacman and not next_ag_pacman:
            reward += ag_carry * 0.15 



        #### OPPONENT PROXIMITY ####
        min_dangerous = util.sys.maxsize
        min_edible = util.sys.maxsize
        
        for opp in opponents:
            opp_state = next_game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()
            
            if opp_pos is None: continue
                
            dist = self.get_maze_distance(next_ag_pos, opp_pos)
            
            if opp_state.scared_timer > 0:
                if dist < min_edible:
                    min_edible = dist
            else:
                if dist < min_dangerous:
                    min_dangerous = dist
    
        # 'min_dangerous_dist' feature: reward for staying safe from dangerous opponents
        if min_dangerous < 5:
            reward -= 0.025 * (5 - min_dangerous)  # Penalty gets worse as distance decreases
        

        # 'min_edible_dist' feature: reward for approaching edible opponents
        if min_edible < 5 and not next_ag_pacman:  # Only when we're ghost
            reward += 0.01 * (5 - min_edible)  # Reward for getting closer

    
        return max(-2.0, min(2.0, reward)) # clamp

    '''

    def get_reward(self, current_state, next_game_state):
        # rewards in range [-2, 2]
        reward = 0.001
        
        # Get current and next states
        ag_state = current_state.get_agent_state(self.index)
        ag_pos = ag_state.get_position()
        ag_pacman = ag_state.is_pacman
        ag_carry = ag_state.num_carrying
        ag_scared = ag_state.scared_timer
        
        next_ag_state = next_game_state.get_agent_state(self.index)
        next_ag_pos = next_ag_state.get_position()
        next_ag_pacman = next_ag_state.is_pacman
        next_ag_carry = next_ag_state.num_carrying
        next_ag_scared = next_ag_state.scared_timer
        
        # for actions (to prevent uselessness)
        if ag_pos == next_ag_pos: reward -= 0.1
        
        # for scoring
        score_diff = self.get_score(next_game_state) - self.get_score(current_state)
        reward += score_diff * 0.5 if score_diff > 0 else 0
        
        # for food
        current_food = self.get_food(current_state).as_list()
        next_food = self.get_food(next_game_state).as_list()
        
        if current_food and next_food:
            reward += 0.3 if len(current_food) > len(next_food) else 0
        
            current_min_food = min([self.get_maze_distance(ag_pos, f) for f in current_food])
            next_min_food = min([self.get_maze_distance(next_ag_pos, f) for f in next_food])
            
            reward += 0.05 if next_min_food < current_min_food else -0.02
        reward += next_ag_carry * 0.01
        
        # for location within the territory
        if next_ag_pacman:
            reward += 0.03 if not ag_pacman else 0

            dist_home = self.get_distance_to_border(next_game_state, next_ag_pos) + 1
            if dist_home > 15:
                reward -= 0.02  # Penalty for being very deep in ur ass 
            elif dist_home > 10:
                reward -= 0.01  # penalty for being deep
            elif next_ag_carry > 0 and  dist_home <= 3:
                reward += 0.02  # can retreat ez
        else: # a la seguent ja és a caseta
            if ag_pacman and ag_carry == 0:
                reward -= 0.1 # si és inutil si havent anat al camp contrari no porta food
        
        # for opponents
        opponents = self.get_opponents(next_game_state)
        closest_dangerous = util.sys.maxsize
        closest_edible = util.sys.maxsize
        
        for opp in opponents:
            opp_state = next_game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()
            
            if opp_pos is None: continue

            dist = self.get_maze_distance(next_ag_pos, opp_pos)
            
            # Big penalty for having ur ass eaten 
            if opp_pos == next_ag_pos:
                reward += 0.6 if opp_state.scared_timer > 0 else -1.2
            
            # Track distances for proximity rewards
            if opp_state.scared_timer > 0:
                if dist < closest_edible: closest_edible = dist
            else:
                if dist < closest_dangerous: closest_dangerous = dist
        
        if closest_dangerous < 5: reward -= (5 - closest_dangerous) * 0.04
        if closest_edible < 7 : reward += (7 - closest_edible) * 0.02
        
        # for lil viagras
        current_caps = self.get_capsules(current_state)
        next_caps = self.get_capsules(next_game_state)
        
        if len(current_caps) > len(next_caps):
            reward += 0.25
            # Strategic bonus
            reward += 0.08*(5 - closest_dangerous) if closest_dangerous < 5 else -0.01*(closest_dangerous - 6)

        if current_caps and next_caps:
            current_min_caps = min([self.get_maze_distance(ag_pos, c) for c in current_caps])
            next_min_caps = min([self.get_maze_distance(next_ag_pos, c) for c in next_caps])
            
            if next_min_caps < current_min_caps:
                reward += 0.02  # Reward for moving toward capsule

        # for dying and being f**kin useless
        if next_ag_pos == self.start: reward -= (1.0 + ag_carry*0.1) 

        # for when it's being a lil pussy
        if next_ag_scared > 0: reward -= 0.01
        
        # for considering time
        food_left = len(self.get_food(next_game_state).as_list())
        total_food_start = 20
        
        # les play it safe
        if food_left <= 4:
            if not next_ag_pacman: reward += 0.05   
        elif food_left < total_food_start * 0.5:  # Less than 50% food left
            # we want violence
            if next_ag_pacman: reward += 0.02
        
        # for team coordination (idk if it does anything but well) 
        # we want one in each state
        teammates = self.get_team(next_game_state)
        teammates.remove(self.index)
        teammate = teammates[0]
        teammate_state = next_game_state.get_agent_state(teammate)
        if next_ag_pacman and not teammate_state.is_pacman: reward += 0.03
        elif not next_ag_pacman and not teammate_state.is_pacman and food_left > total_food_start * 0.5: reward -= 0.02
        # bitch is slacking

        
        # Clip to prevent extreme values that could destabilize learning (⌐⊙_⊙)
        reward = max(-2.0, min(2.0, reward)) # tho if everything works, this shouldn't be necessary...
        
        # if random.random() < 0.01 and IS_TRAINING:  # 1% of steps
        #     print(f"\nReward for step {self.current_iteration}:")
        #     print(f"  Final reward: {reward:.3f}")
        #     print(f"  Position: {next_ag_pos}, Pacman: {next_ag_pacman}, Carrying: {next_ag_carry}")
        #     print(f"  Closest dangerous: {closest_dangerous if closest_dangerous < float('inf') else 'N/A'}")
        #     print(f"  Closest edible: {closest_edible if closest_edible < float('inf') else 'N/A'}")
        
        return reward

'''
class DefensiveQLearner(ApproxQLearningAgent) :

    def __init__(self, index, time_for_computing=.1) :
        super().__init__(index, time_for_computing)
        self.weights = {
            'score' : 1.,
            'bias' : 1.,
            'scared_timer' : 1.,
            'opp_1_far' : 1.,
            'opp_2_far' : 1.,
            'min_opp_dist' : 1.,
            'food_to_defend' : 1.,
            'most_carrying_opp_dist' : 1.
            }

    def get_features(self, game_state, action) :

        features = util.Counter()

        successor = self.get_successor(game_state, action)
        pos = successor.get_agent_state(self.index).get_position()

        #Food to defend left
        food_list = self.get_food_you_are_defending(successor).as_list()
        features['food_to_defend'] = len(food_list)

        #Are you scared?
        features['scared_timer'] = successor.get_agent_state(self.index).scared_timer

        #Opponents (A lot of stuff Rita wants as features, some I don't know how to name properly)
        opponents = self.get_opponents(game_state)

        min_opp_dist = 9999
        max_carry = -9999
        
        for opp in opponents :
            
            opp_state = successor.get_agent_state(opp)
            opp_pos = opp_state.get_position()
            carrying = opp_state.num_carrying
            
            dist = self.get_maze_distance(opp_pos, pos) if opp_pos != None else successor.get_agent_distances()[opp]

            if carrying > max_carry : features['most_carrying_opp_dist'] = dist
            
            if min_opp_dist > dist : min_opp_dist = dist
            
            if dist < 2 : features['opp_1_far'] += 1

            elif dist == 2 : features['opp_2_far'] += 1

        features['min_opp_dist'] = min_opp_dist

        #score, simple enough
        features['score'] = self.get_score(successor)
        
        #bias
        features['bias'] = 1.
        
        return features
    

    
    def get_reward(self, current_game_state, next_game_state):
        
        reward = 0

        new_food_list = self.get_food(next_game_state).as_list()
        old_food_list = self.get_food(current_game_state).as_list()

        if len(new_food_list) < len(old_food_list) : reward += 10

        old_score = self.get_score(current_game_state)
        new_score = self.get_score(next_game_state)
        
        if new_score > old_score : reward += (new_score - old_score)*10

        #Killed somebody? good
        opponents = self.get_opponents()

        for O_idx in opponents :
            O_state = next_game_state.get_agent_state(O_idx)
            if O_state.get_position() == O_state.start : reward += 100

        #Dead? useless
        if next_game_state.get_agent_state(self.index).get_position() == self.start : reward -= 100

        return reward

        pass
'''

class DefensiveQLearner(ApproxQLearningAgent):
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        
        if IS_TRAINING:
            try:
                with open("weights_defensive.json", "r") as f:
                    loaded_weights = json.load(f)
                    self.weights = util.Counter(loaded_weights)
            except:
                self.initialize_weights_random()
        else:
            # Testing weights - initialize with sensible defaults
            self.initialize_weights()
        
        self.last_invader_positions = []
        self.patrol_points = []
        self.choke_points = []

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        

        self.identify_strategic_points(game_state)
        
        self.invader_history = []
        self.last_patrol_point = None

    def initialize_weights(self):
        self.weights = util.Counter({
            "bias": 0.022608033328426466, 
            "moved": 0.10909000196826332, 
            "is_stop_action": -0.19151577969322647, 
            "action_reverse": -0.024973774944258287, 
            "invader_present": 0.3153991823049016, 
            "invader_distance": -0.17006177188821145, 
            "closest_invader_dist": -0.1450679218773895, 
            "invader_1_step": 0.4255339442468183, 
            "invader_2_step": 0.4064839944435447, 
            "invader_3_plus_step": 0.0750115167672083, 
            "invader_carrying_food": 0.2879283750820346, 
            "multiple_invaders": 0.37130989009938564, 
            "food_defended": 0.18622204134849257, 
            "food_being_eaten": -0.12972746009442462, 
            "min_food_distance": 0.10636334403949122, 
            "food_density": 0.10589736979228564, 
            "patrol_score": 0.09647715485098493, 
            "choke_point_control": 0.22747180153783578, 
            "center_position": 0.15239097356731784, 
            "flank_protection": 0.038792946252106796, 
            "eating_invader": 0.7575613775318564, 
            "near_edible_opponent": 0.31209507401417075, 
            "near_dangerous_opponent": -0.1586623948120628, 
            "im_scared": -0.31028556523749976, 
            "scared_timer": -0.060514473897994005, 
            "border_distance": -0.10372056134517002, 
            "teammate_distance": 0.019807491120540758, 
            "teammate_on_defense": 0.22214844508978251, 
            "we_scared": -0.128905785702187804, 
            "predictive_positioning": 0.14798791327220749, 
            "time_pressure": 0.045824034339444})

    def initialize_weights_random(self):
        """Initialize weights for defensive behavior"""
        self.weights = util.Counter({
            # === BASIC ACTION WEIGHTS ===
            'bias': 0.05,                     # Positive bias for action
            'moved': 0.12,                    # Strong reward for movement (patrol)
            'is_stop_action': -0.4,           # Very strong penalty for stopping
            'action_reverse': -0.15,          # Penalize backtracking
            
            # === INVADER DETECTION & RESPONSE ===
            'invader_present': 0.3,           # Positive when invaders exist
            'invader_distance': -0.2,         # NEGATIVE: closer invaders = more urgent
            'closest_invader_dist': -0.25,    # NEGATIVE: closer = more important
            'invader_1_step': 0.4,            # Big reward for very close invader
            'invader_2_step': 0.2,            # Good reward for nearby invader
            'invader_3_plus_step': 0.05,      # Small reward for distant invader
            
            # === INVADER CHARACTERISTICS ===
            'invader_carrying_food': 0.35,    # Prioritize invaders with food
            'multiple_invaders': 0.4,         # Multiple invaders = more urgent
            
            # === FOOD PROTECTION ===
            'food_defended': 0.15,            # Value of remaining food
            'food_being_eaten': -0.3,         # Penalty when food is actively being eaten
            'min_food_distance': 0.08,        # Positive: closer to food clusters = better
            'food_density': 0.1,              # Protect dense food areas
            
            # === PATROL & POSITIONING ===
            'patrol_score': 0.1,              # Reward for good patrol patterns
            'choke_point_control': 0.25,      # High value for controlling choke points
            'center_position': 0.15,          # Good to be centrally located
            'flank_protection': 0.1,          # Protect vulnerable flanks
            
            # === OPPONENT INTERACTION ===
            'eating_invader': 0.8,            # BIG reward for eating invader
            'near_edible_opponent': 0.25,     # Good to be near scared opponents
            'near_dangerous_opponent': -0.2,  # Bad to be near dangerous opponents
            
            # === SAFETY & STATE ===
            'im_scared': -0.3,                # Negative when scared
            'scared_timer': -0.15,            # More scared time = worse
            'border_distance': -0.1,          # Negative: closer to border = riskier
            
            # === TEAM COORDINATION ===
            'teammate_distance': 0.05,        # Good to be near teammate for support
            'teammate_on_defense': 0.1,       # Good if teammate is also defending
            
            # === STRATEGIC AWARENESS ===
            'we_scared': -0.25,    # Bad when enemy has capsules
            'predictive_positioning': 0.12,   # Reward for anticipating invader paths
            'time_pressure': 0.05,            # More aggressive when winning
        })
        
        # Add small random variations
        for feature in self.weights:
            self.weights[feature] *= random.uniform(0.9, 1.1)
    

    # Precompute patrol points and choke points: important defensive positions
    def identify_strategic_points(self, game_state):
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        midpoint = width // 2
        
        our_side = range(0, midpoint) if self.red else range(midpoint, width)
        
        self.patrol_points = []
        for x in our_side:
            for y in range(1, height-1):
                if not walls[x][y]:
                    # Check if it's not too close to border
                    border_dist = abs(x - midpoint)
                    if border_dist > 2: self.patrol_points.append((x, y))
        
        # Identify choke points (narrow passages)
        self.choke_points = []
        for x in our_side:
            for y in range(1, height-1):
                if not walls[x][y]:
                    open_neighbors = 0
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]: open_neighbors += 1
                    
                    if open_neighbors <= 2:
                        self.choke_points.append((x, y))
        
        if not self.choke_points and self.patrol_points:
            xs = [p[0] for p in self.patrol_points]
            ys = [p[1] for p in self.patrol_points]
            center_x = sum(xs) // len(xs)
            center_y = sum(ys) // len(ys)
            self.choke_points.append((center_x, center_y))
    
    def get_features(self, game_state, action):
        features = util.Counter()
        
        ag_state = game_state.get_agent_state(self.index)
        ag_pos = ag_state.get_position()
        
        
        next_game_state = self.get_successor(game_state, action)
        next_ag_state = next_game_state.get_agent_state(self.index)
        next_ag_pos = next_ag_state.get_position()
        next_ag_scared = next_ag_state.scared_timer
                
        features['bias'] = 1.0
        features['moved'] = 1.0 if ag_pos != next_ag_pos else 0.0
        features['is_stop_action'] = 1.0 if action == 'Stop' else 0.0
        
        previous_game_state = self.get_previous_observation()
        if previous_game_state is not None:
            previous_actions = previous_game_state.get_legal_actions(self.index)
            for action in previous_actions:
                if self.get_successor(previous_game_state, action) == game_state:
                    # print("found previous gs and action")
                    rev = Directions.REVERSE.get(action, None)
                    features['action_reverse'] = 1.0 if action == rev else 0.0
                    break

        # if hasattr(self, 'last_action') and self.last_action:
        #     rev = Directions.REVERSE.get(self.last_action, None)
        #     features['action_reverse'] = 1.0 if action == rev else 0.0
        # self.last_action = action
                
        opponents = [[i, next_game_state.get_agent_state(i)] for i in self.get_opponents(next_game_state)]
        invaders = [o for o in opponents if o[1].is_pacman]
        
        # for op in opponents:

        # dist = self.get_maze_distance(opp_pos, next_ag_pos) if opp_pos != None else next_game_state.get_agent_distances()[opp]


        # Basic invader presence
        features['invader_present'] = 1.0 if invaders else 0.0
        features['multiple_invaders'] = min(1.0, len(invaders) / 2.0)
        
        # per si de cas:
        features['invader_1_step'] = 0.0
        features['invader_2_step'] = 0.0
        features['invader_3_plus_step'] = 0.0
        features['invader_carrying_food'] = 0.0


        if invaders:
            invader_dists = []
            closest_invader = None
            min_invader_dist = util.sys.maxsize
            total_food_carried = 0
            
            for i in invaders:
                invader_pos = i[1].get_position()
                dist = self.get_maze_distance(invader_pos, next_ag_pos) if invader_pos != None else next_game_state.get_agent_distances()[i[0]]
                invader_dists.append(dist)
                
                if dist < min_invader_dist:
                    min_invader_dist = dist
                    closest_invader = i[1]
                
                total_food_carried += i[1].num_carrying
            
            features['invader_distance'] = 1.0 / (1.0 + min_invader_dist / 10.0) 
            features['closest_invader_dist'] = 1.0 / (1.0 + min_invader_dist / 5.0)
            
            if min_invader_dist == 1:
                features['invader_1_step'] = 1.0
            elif min_invader_dist == 2:
                features['invader_2_step'] = 1.0
            elif min_invader_dist >= 3:
                features['invader_3_plus_step'] = min(1.0, 10.0 / min_invader_dist)
            
            if closest_invader:
                c = closest_invader.num_carrying
                features['invader_carrying_food'] = 2 - (1/(c + 1)**0.5) - (1/c**0.08) if c > 0. else 0.
        else:
            # No invaders
            features['invader_distance'] = 0.0
            features['closest_invader_dist'] = 0.0
        
        
        next_food = self.get_food_you_are_defending(next_game_state).as_list()
        current_food = self.get_food_you_are_defending(game_state).as_list()

        features['food_defended'] = min(1.0, len(next_food) / 20.0)
        
        # is some sneaky b*tch eating our treasure?  
        features['food_being_eaten'] = (len(current_food) - len(next_food))/2.

        if next_food:
            food_dists = [self.get_maze_distance(next_ag_pos, f) for f in next_food]
            features['min_food_distance'] = 10.0 / (10.0 + min(food_dists))
            
            if len(next_food) > 1:
                total_dist = 0
                count = 0
                for i in range(len(next_food)):
                    for j in range(i+1, len(next_food)):
                        total_dist += self.get_maze_distance(next_food[i], next_food[j])
                        count += 1

                avg_dist = total_dist / count
                features['food_density'] = 20.0 / (20.0 + avg_dist)  # Av dist smaller => closer => denser
        else:
            features['min_food_distance'] = 0.0
            features['food_density'] = 0.0
        


        # encourage patrolling bc what else to do ...
        features['patrol_score'] = self.get_patrol_score(next_ag_pos, game_state)
        
        # funfact: Jimi Hendrix died from choking on his own vomit while unconcious with barbiturates. at the age of 27 :)
        if self.choke_points:
            choke_dists = [self.get_maze_distance(next_ag_pos, cp) for cp in self.choke_points]
            min_choke_dist = min(choke_dists)
            features['choke_point_control'] = 10.0 / (10.0 + min_choke_dist)
        else:
            features['choke_point_control'] = 0.0
        
        walls = next_game_state.get_walls()
        center_x = walls.width // 2
        center_y = walls.height // 2
        dist_to_center = self.get_maze_distance(next_ag_pos, (center_x, center_y))
        features['center_position'] = 20.0 / (20.0 + dist_to_center)
        
        features['flank_protection'] = self.get_flank_protection_score(next_ag_pos, next_game_state)
        
        
        # Check if we are chomping an invader eheheheeh
        for invader in invaders:
            features['eating_invader'] = 1.0 if invader[1].get_position() == next_ag_pos else 0.0
            break
        
        # Proximity to yummy opponents 
        yummy_opponents = [o for o in invaders if not next_ag_scared]
        dist = [next_game_state.get_agent_distances()[i[0]] for i in yummy_opponents]
        features['near_edible_opponent'] = 10.0 / (10.0 + min(dist)) if dist else 0.0
        
        # run mf run (only for the enemies taht are near tho...)
        dangerous = [o for o in opponents if o[1].get_position() is not None and next_ag_scared]
        if dangerous:
            danger_dists = [self.get_maze_distance(next_ag_pos, i[1].get_position()) for i in dangerous]
            features['near_dangerous_opponent'] = 10.0 / (10.0 + min(danger_dists))
        else:
            features['near_dangerous_opponent'] = 0.0
        
        
        features['im_scared'] = 1.0 if next_ag_scared else 0.0
        features['scared_timer'] = min(1.0, next_ag_scared/ 40.0)  # 0-1 scaled
        
        border_dist = self.get_distance_to_border(next_game_state, next_ag_pos)
        features['border_distance'] = max(0. , min(border_dist / 20.0, 1.))
        

        teammates = self.get_team(next_game_state)
        teammates.remove(self.index)
        teammate = teammates[0]
        teammate_state = next_game_state.get_agent_state(teammate)
        teammate_dist = next_game_state.get_agent_distances()[teammate]
        
        features['teammate_distance'] = 10.0 / (10.0 + teammate_dist)
        features['teammate_on_defense'] = 1.0 if not teammate_state.is_pacman else 0.0
        
        # viagra status
        # capsules = self.get_capsules(next_game_state)

        features['we_scared'] = 1.0 if next_ag_scared else 0.0        
        features['predictive_positioning'] = self.get_predictive_score(next_ag_pos, invaders, next_game_state)
        
        # more aggressive if winning
        our_score = self.get_score(next_game_state)
        features['time_pressure'] = min(1.0, max(0.0, our_score / 10.0))
        
        return features
    
    def get_patrol_score(self, position, game_state):
        if not hasattr(self, 'visited_positions'):
            self.visited_positions = []
        
        self.visited_positions.append(position)
        if len(self.visited_positions) > 20: self.visited_positions.pop(0)
        if len(self.visited_positions) < 2: return 0.0
        
        recent_positions = self.visited_positions[-5:] if len(self.visited_positions) >= 5 else self.visited_positions
        min_recent_dist = min([self.get_maze_distance(position, p) for p in recent_positions[:-1]])
        
        # Higher score for being further from recently visited spots
        # we don't want you biting your own ass duh 
        patrol_score = min(1.0, min_recent_dist / 10.0)
        
        # Bonus for patrolling interesting points
        if self.choke_points:
            choke_dists = [self.get_maze_distance(position, cp) for cp in self.choke_points]
            if min(choke_dists) <= 2: patrol_score += 0.2
        
        return min(1.0, patrol_score)
    
    # fun fact: Sun Tzu's The Art of War strongly emphasizes the use of flanking, although it does not advocate completely surrounding
    # the enemy force as this may induce it to fight with greater ferocity if it cannot escape.
    #~~~~~~ I will win but never fight ~~~~~~#
    #~~~~~~~~ THAT'S THE  ART OF WAR ~~~~~~~~#
    def get_flank_protection_score(self, position, game_state):
        walls = game_state.get_walls()
        vulnerable_count = 0
        
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = int(position[0] + dx), int(position[1] + dy)
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                # Check if this way leads to food
                food_list = self.get_food_you_are_defending(game_state).as_list()
                if food_list:
                    for food_pos in food_list:
                        if (dx > 0 and food_pos[0] > position[0]) or (dx < 0 and food_pos[0] < position[0]) or \
                           (dy > 0 and food_pos[1] > position[1]) or (dy < 0 and food_pos[1] < position[1]):
                            vulnerable_count += 0.25
                            break
        
        return min(1.0, vulnerable_count)
    

    def get_predictive_score(self, position, invaders, game_state):
        if not invaders: return 0.0
        current_invader_positions = []
        for i in invaders:
            i_pos = i[1].get_position() 
            if i_pos is not None:
                current_invader_positions.append(i_pos)
        if not current_invader_positions:return 0.0
        # invader positions over time
        # current_invader_distance = [game_state.get_agent_distances()[inv[0]] for inv in invaders]

        self.invader_history.append(current_invader_positions)
        if len(self.invader_history) > 5: self.invader_history.pop(0)
        if len(self.invader_history) >= 2:
            # invaders like to move it, move it (twds the food :v)
            food_list = self.get_food_you_are_defending(game_state).as_list()
            threatened_food = []
            for food_pos in food_list:
                for inv_pos in current_invader_positions:
                    dist = self.get_maze_distance(inv_pos, food_pos)
                    threatened_food.append((food_pos, dist))
            
            # Sort by threat level 
            threatened_food.sort(key=lambda x: x[1])
            most_threatened_food = threatened_food[0][0]
            
            dist_to_threatened = self.get_maze_distance(position, most_threatened_food)
            return 10.0 / (10.0 + dist_to_threatened)
        
        return 0.0
    
    def get_reward(self, current_state, next_game_state):
        reward = -0.001
        
        ag_state = current_state.get_agent_state(self.index)
        next_ag_state = next_game_state.get_agent_state(self.index)
        next_ag_pos = next_ag_state.get_position()
        
        if ag_state.get_position() == next_ag_pos: reward -= 0.08
                
        current_opponents = [[i, current_state.get_agent_state(i)] for i in self.get_opponents(current_state)]
        next_opponents = [[i, next_game_state.get_agent_state(i)] for i in self.get_opponents(next_game_state)]
        
        current_invaders = [e for e in current_opponents if e[1].is_pacman]
        next_invaders = [e for e in next_opponents if e[1].is_pacman]
        
        # Big fat juicy reward for eating an invader
        for invader in next_invaders:
            if invader[1].get_position() == next_ag_pos:
                reward += 0.8 
                # bonus if he was full ;)
                if invader.num_carrying > 0: reward += invader.num_carrying * 0.15
        
        if len(next_invaders) < len(current_invaders): reward += 0.1
        if len(next_invaders) > len(current_invaders): reward -= 0.15

        invader_dists = []
        if next_invaders:
            for i in next_invaders:
                i_pos = i[1].get_position()
                dist = self.get_maze_distance(i_pos, next_ag_pos) if i_pos != None else next_game_state.get_agent_distances()[i[0]]
                invader_dists.append(dist)
            min_invader_dist = min(invader_dists)
            if min_invader_dist <= 3: reward += 0.05 * (4 - min_invader_dist)

        
        current_food = len(self.get_food_you_are_defending(current_state).as_list())
        next_food = len(self.get_food_you_are_defending(next_game_state).as_list())
        
        if next_food < current_food:
            food_lost = current_food - next_food
            reward -= food_lost * 0.3 
        
        if next_invaders and next_food == current_food: reward += 0.05
                
        # patrolling choke points
        if self.choke_points:
            choke_dists = [self.get_maze_distance(next_ag_pos, cp) for cp in self.choke_points]
            if min(choke_dists) <= 2: reward += 0.03 
        
        # Penalty for staying far from food 
        if not next_invaders:
            food_list = self.get_food_you_are_defending(next_game_state).as_list()
            food_dists = [self.get_maze_distance(next_ag_pos, f) for f in food_list]
            avg_food_dist = sum(food_dists) / len(food_dists)
            if avg_food_dist > 10: reward -= 0.02
                
        if next_ag_state.scared_timer > 0: reward -= 0.01
        if next_ag_pos == self.start: reward -= 0. # died
        
        
        # Reward for team coordination
        teammates = self.get_team(next_game_state)
        teammates.remove(self.index)
        teammate = teammates[0]
        
        teammate_state = next_game_state.get_agent_state(teammate)
        if not next_ag_state.is_pacman and teammate_state.is_pacman: reward += 0.03
        
        food_left = len(self.get_food(next_game_state).as_list())
        total_food_start = 20
        
        if food_left <= 2: reward += 0.05
        elif food_left < total_food_start * 0.3:  # Less than 30% food left
            if not next_invaders: reward += 0.02
        
        
        reward = max(-2.0, min(2.0, reward))
        
        # Debug output (occasional)
        # if random.random() < 0.001 and IS_TRAINING:
        #     print(f"\nDefensive Agent {self.index} Reward:")
        #     print(f"  Final: {reward:.3f}")
        #     print(f"  Invaders: {len(next_invaders)}, Food protected: {next_food}/{current_food}")
        #     print(f"  Position: {next_ag_pos}, Scared: {next_ag_state.scared_timer > 0}")
        
        return reward
    
    def final(self, game_state):
        """Save defensive-specific weights"""
        super().final(game_state)
        
        try:
            with open("weights_defensive.json", "w") as f:
                json.dump(dict(self.weights), f)
        except:
            pass



class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
    """

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        opponents = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in opponents if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
