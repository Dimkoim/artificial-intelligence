from sample_players import DataPlayer
import math, random


class CustomPlayer_AB(DataPlayer):

    def minimax(self, state, depth):
        '''
        Min-max algorithm
        :param state: Game state
        :param depth: Depth of tree
        :return: the state with highest score
        '''
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def alphabeta(self, state, depth):
        '''
         Return the move along a branch of the game tree that
         has the best possible value.  A move is a pair of coordinates
         in (column, row) order corresponding to a legal move for
         the searching player.
        '''
        def min_value(state, depth, alpha, beta):
            '''
            Return the value for a win (+1) if the game is over,
            otherwise return the minimum value over all legal child
            nodes
            '''
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha: break
            return value

        def max_value(state, depth, alpha, beta):
            '''
            Return the value for a loss (-1) if the game is over,
            otherwise return the maximum value over all legal child
            nodes.
            '''
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha: break
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, float("-inf"), float("inf")))

    def score(self, state):
        '''
        Scoring function
        '''
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def get_action(self, state):
        import random
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            depth = 3
            # Iterative Deepening
            for i in range(1, depth + 1):
                self.queue.put(self.alphabeta(state, depth=i))


class MCTS():
    def __init__(self, state):
        self.root_node = self.TreeNode(state)

    def select(self, node):
        while not node.state.terminal_test():
            if not node.explored():
                expand_node = self.expand(node)
                return expand_node
            else:
                node = self.best_child(node)
        return node

    def best_child(self, node):
        '''
        Select a unexplored child, or best child

        '''
        best_child_nodes = []
        best_score = float('-inf')

        C = math.sqrt(2)
        for child in node.childrens:
            exploit = child.q_value / child.visited
            explore = C * math.sqrt(math.log(node.visited) / child.visited)
            child_score = exploit + explore
            if child_score == best_score:
                best_child_nodes.append(child)
            elif child_score > best_score:
                best_child_nodes = []
                best_child_nodes.append(child)
                best_score = child_score

        if len(best_child_nodes) == 0:
            return None
        return random.choice(best_child_nodes)

    def expand(self, node):
        '''
        Pick an action, execute and get next child
        '''
        possible_actions = node.actions_available()

        if len(possible_actions) > 0:
            action = possible_actions[0]
            child_state = node.state.result(action)
            child_node = MCTS.TreeNode(child_state, node, action)
            node.childrens.append(child_node)
            node.actioned.append(action)

            return node.childrens[-1]
        else:
            return None

    def simulate(self, state):
        '''
        Simulate to the end of the game, get reward 1 in case of winning of -1 otherwise
        '''
        player_id = state.player()
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
        return -1 if state._has_liberties(player_id) else 1

    def backpropagation(self, node, reward):
        '''
        Update all nodes with reward, from leaf node all the way back to the root
        '''
        while node is not None:
            node.update_qvalue(reward)
            node = node.parent
            reward = -reward

    def best_action(self, node):
        return self.best_child(node).parent_action

    def run(self):
        num_iter = 60
        try:
            if self.root_node.state.terminal_test():
                return random.choice(self.root_node.state.actions())
            for i in range(num_iter):
                node = self.select(self.root_node)
                if node is None:
                    continue
                reward = self.simulate(node.state)
                self.backpropagation(node, reward)
        except Exception as ex:
            print('Exception: {0}'.format(str(ex)))

        action = self.best_action(self.root_node)

        return action

    class TreeNode():
        '''
        Represents a game state, with available actions to explore future game states further down the tree
        '''
        def __init__(self, state, parent=None, parent_action=None):
            self.state = state
            self.parent = parent
            self.parent_action = parent_action
            self.actions = state.actions()
            self.actioned = []
            self.childrens = []
            self.q_value = 0
            self.visited = 1

        def explored(self):
            return len(self.actions) == len(self.actioned)

        def actions_available(self):
            actions_left = list(set(self.actions) - set(self.actioned))
            return actions_left

        def update_qvalue(self, reward):
            self.q_value += reward
            self.visited += 1


class CustomPlayer_MCTS(DataPlayer):
    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            mcts = MCTS(state)
            best_move = mcts.run()
            self.queue.put(best_move)


CustomPlayer = CustomPlayer_MCTS