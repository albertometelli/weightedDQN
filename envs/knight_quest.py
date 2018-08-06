import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.envs.registration import register


MAP = [
    "+-------+",
    "|P: : :T|",
    "| : : : |",
    "| : : : |",
    "| :G: : |",
    "+-------+",
]

register(
    id='KnightQuest-v0',
    entry_point='envs.knight_quest:KnightQuest',
)

class KnightQuest(DiscreteEnv):
    def __init__(self,
                 armor_move_prob=0.5,
                 armor_collect_prob=0.01):
        self.desc = np.asarray(MAP, dtype='c')

        self.action_names = np.array(['right', 'down', 'left', 'up', 'collect', 'buy_pass', 'buy_armor', 'stay'])

        self.enemy_locs = [(0,1), (1,0), (1,1)]
        self.goldloc = (3,1)
        self.townloc = (0,3)
        self.target = (0,0)

        self.n_rows, self.n_cols = 4, 4
        self.n_objects = 4
        self.n_golds = 2
        maxR = self.n_rows - 1
        maxC = self.n_cols - 1

        self.armor_collect_prob = armor_collect_prob
        self.armor_move_prob = armor_move_prob

        nA = len(self.action_names)

        isd = []
        for enemypos in range(3):
            state = self.encode(self.townloc[0], self.townloc[1], 0, enemypos, 0)
            isd += [(state, 1./3.)]

        fromExtendedToCompactIdxs = {}
        fromCompactToExtended = []
        absorbing_states = []

        real_nS = -1
        P = {}
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                for goldlevel in range(self.n_golds):
                    for enemypos in range(len(self.enemy_locs)):
                        for object in range(self.n_objects):
                            state = self.encode(row, col, goldlevel, enemypos, object)

                            if  self.is_absorbing_state(row, col, goldlevel, enemypos, object):
                                real_nS += 1
                                P[real_nS] = {a : [] for a in range(nA)}
                                fromExtendedToCompactIdxs[state] = real_nS
                                fromCompactToExtended.append(state)
                                absorbing_states.append(real_nS)
                                continue
                            else:
                                real_nS += 1
                                P[real_nS] = {a : [] for a in range(nA)}
                                fromExtendedToCompactIdxs[state] = real_nS
                                fromCompactToExtended.append(state)

                                if enemypos == 0:
                                    enemypossiblelocs = [(0,0.4), (2,0.6)]
                                elif enemypos == 1:
                                    enemypossiblelocs = [(1,0.4), (2,0.6)]
                                else:
                                    enemypossiblelocs = [(0,0.4), (1,0.2), (2, 0.4)]
                                for newenemypos, newenemyposprob in enemypossiblelocs:
                                    for a in range(nA):
                                        # defaults
                                        newrow, newcol = row, col
                                        newgoldlevel, newobject = goldlevel, object
                                        reward = -1
                                        agentloc = (row,col)

                                        if object < 2:
                                            if a == 0:
                                                newcol = min(col + 1, maxC)
                                            elif a == 1:
                                                newrow = min(row + 1, maxR)
                                            elif a == 2:
                                                newcol = max(col - 1, 0)
                                            elif a == 3:
                                                newrow = max(row - 1, 0)
                                            elif a == 4:  # collect
                                                if (agentloc == self.goldloc):
                                                    newgoldlevel = min(goldlevel+1, self.n_golds - 1)
                                                else:
                                                    reward = -10
                                            elif a == 5:  # buy_pass
                                                if (agentloc == self.townloc) and goldlevel == 1:
                                                    newgoldlevel = 0
                                                    newobject = 1 if object in [0,1] else 3
                                                else:
                                                    reward = -10
                                            elif a == 6: # buy armor
                                                if (agentloc == self.townloc) and goldlevel == 1:
                                                    newgoldlevel = 0
                                                    newobject = 2 if object in [0,2] else 3
                                                else:
                                                    reward = -10
                                            nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos, newobject)
                                            
                                            if (newrow, newcol) == self.enemy_locs[newenemypos] and newobject not in [2,3]:
                                                # agent moves in the same position of the enemy
                                                # but she does not have the armor => reset
                                                reward = -20
                                                P[real_nS][a] += [(nextstate,  newenemyposprob, reward)]
                                                    # P[state, a, state_id] = prob * newenemyposprob
                                            elif (newrow, newcol) == self.target and newobject in [1,3]:
                                                # the agent reaches the target (princess) and has the key => reset
                                                reward = 20
                                                P[real_nS][a] += [(nextstate,  newenemyposprob, reward)]
                                                    # P[state, a, state_id] = prob * newenemyposprob
                                            else:
                                                # normal transition
                                                if not self.is_absorbing_state(newrow, newcol, newgoldlevel, newenemypos, newobject):
                                                    nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos, newobject)
                                                    P[real_nS][a] += [(nextstate, newenemyposprob, reward)]
                                                    # P[state, a, nextstate] += newenemyposprob
                                        else:
                                            agent_pos_prob = []
                                            # agent has the armor
                                            if a == 0:
                                                agent_pos_prob = [(newrow, min(col + 1, maxC), armor_move_prob),
                                                                  (newrow, col, 1-armor_move_prob)]
                                            elif a == 1:
                                                agent_pos_prob = [(min(row + 1, maxR), newcol, armor_move_prob),
                                                                  (row, newcol, 1-armor_move_prob)]
                                            elif a == 2:
                                                agent_pos_prob = [(newrow, max(col - 1, 0), armor_move_prob),
                                                                  (newrow, col, 1-armor_move_prob)]
                                            elif a == 3:
                                                agent_pos_prob = [(max(row - 1, 0), newcol, armor_move_prob),
                                                                  (row, newcol, 1-armor_move_prob)]
                                            elif a == 4:  # collect
                                                if (agentloc == self.goldloc):
                                                    for newgoldlevel, p in [(min(goldlevel+1, self.n_golds - 1), armor_collect_prob),
                                                                            (goldlevel, 1-armor_collect_prob)]:
                                                        nextstate = self.encode(newrow,newcol,newgoldlevel,newenemypos,newobject)
                                                        P[real_nS][a] += [(nextstate, newenemyposprob * p, -1)]
                                                        # P[state, a, nextstate] += newenemyposprob * p
                                                else:
                                                    reward = -10
                                                    P[real_nS][a] += [(state, newenemyposprob, reward)]
                                                    # P[state, a, state] += newenemyposprob
                                            elif a == 5:  # buy_pass
                                                if (agentloc == self.townloc) and goldlevel == 1:
                                                    newgoldlevel = 0
                                                    newobject = 1 if object in [0,1] else 3
                                                else:
                                                    reward = -10

                                                nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos,
                                                                        newobject)
                                                P[real_nS][a] += [(nextstate, newenemyposprob, reward)]
                                                # P[state, a, nextstate] += newenemyposprob
                                            elif a == 6: # buy armor
                                                if (agentloc == self.townloc) and goldlevel == 1:
                                                    newgoldlevel = 0
                                                    newobject = 2 if object in [0,2] else 3
                                                else:
                                                    reward = -10

                                                nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos,
                                                                        newobject)
                                                P[real_nS][a] += [(nextstate, newenemyposprob, reward)]
                                                # P[state, a, nextstate] += newenemyposprob
                                            else: #stay
                                                agent_pos_prob = [(row, col, 1.)]

                                            if a in [0,1,2,3,7]:
                                                for newrow, newcol, p in agent_pos_prob:
                                                    if (newrow, newcol) == self.enemy_locs[newenemypos] and newobject not in [2, 3]:
                                                        # agent moves in the same position of the enemy
                                                        # but she does not have the armor => reset
                                                        reward = -20
                                                        nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos,
                                                                        newobject)
                                                        P[real_nS][a] += [(nextstate,  p * newenemyposprob, reward)]
                                                            # P[state, a, state_id] += prob * p
                                                    elif (newrow, newcol) == self.target and newobject in [1, 3]:
                                                        # the agent reaches the target (princess) and has the key => reset
                                                        reward = 20
                                                        nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos,
                                                                        newobject)
                                                        P[real_nS][a] += [(nextstate, p * newenemyposprob, reward)]
                                                            # P[state, a, state_id] += prob * p
                                                    else:
                                                        # normal transition
                                                        reward = -1
                                                        if not self.is_absorbing_state(newrow, newcol, newgoldlevel, newenemypos,newobject):
                                                            nextstate = self.encode(newrow, newcol, newgoldlevel, newenemypos, newobject)
                                                            P[real_nS][a] += [(nextstate, newenemyposprob * p, reward)]
                                                            # P[state, a, nextstate] += newenemyposprob * p

        real_nS += 1

        self.isd = []
        for s, p in isd:
            self.isd.append((fromExtendedToCompactIdxs[s], p))
        self.P_mat = np.zeros((real_nS, nA, real_nS))
        self.R_mat = np.zeros((real_nS, nA, real_nS))

        self.fromExtendedToCompactIdxs = fromExtendedToCompactIdxs
        self.fromCompactToExtended = fromCompactToExtended
        self.absorbing_states=absorbing_states
        state_actions = []
        for s in range(real_nS):
            for a in range(nA):
                # print(s, self.fromCompactToExtended[s], self.decode(self.fromCompactToExtended[s]), a)
                tot = 0.
                L = {}
                expected_r = 0
                for next_state, p, reward in P[s][a]:
                    next = fromExtendedToCompactIdxs[next_state]
                    if next not in L.keys():
                        L[next] = p
                    else:
                        L[next] = L[next] + p
                    self.P_mat[s,a,next] += p
                    self.R_mat[s,a, next] += reward
                    expected_r += p * reward
                    tot += p
                #assert np.isclose(tot, 1.), tot
                P[s][a] = ([],[])
                for k, v in L.items():
                    P[s][a][0].append(k)
                    P[s][a][1].append(v)
            state_actions.append(list(range(nA)))
        
        # self.P = P
        #self.R_mat = (self.R_mat + 20) / 40.
        
        super(KnightQuest, self).__init__(real_nS, nA, P, isd)

        self.initial_state = 0
        self.state = 0  # Index of the current state
        self.state_actions = state_actions
        self.max_nb_actions_per_state = max(map(len, self.state_actions))  # Maximal number of actions per state
        self.reward = None  # Last reward obtained while exploring the SMDP (init = 0)
        self.holding_time = None  # Last holding time obtained while exploring the SMDP (init = 0)
        self.nb_states = len(self.state_actions)  # Total number of states
        self.max_nb_actions_per_state = max(map(len, self.state_actions))

        self.reset()
        self.lastaction = None
        self.holding_time = 1
        
    def is_absorbing_state(self, agentrow, agentcol, goldlevel, enemypos, object):
        if (agentrow, agentcol) == self.enemy_locs[enemypos] and object not in [2, 3] \
            or (agentrow, agentcol) == self.target and object in [1, 3]:
            return True
        return False

    def encode(self, agentrow, agentcol, goldlevel, enemypos, object):
        # nr, nc, n_enemyloc, n_objects
        # http://www.alecjacobson.com/weblog/?p=1425
        le = len(self.enemy_locs)
        i = object
        i *= le
        i += enemypos
        i *= self.n_golds
        i += goldlevel
        i *= self.n_cols
        i += agentcol
        i *= self.n_rows
        i += agentrow
        return i

    def decode(self, i):
        # http://www.alecjacobson.com/weblog/?p=1425
        out = []
        le = len(self.enemy_locs)
        out.append(i % self.n_rows)
        i = i // self.n_rows
        out.append(i % self.n_cols)
        i = i // self.n_cols
        out.append(i % self.n_golds)
        i = i // self.n_golds
        out.append(i % le)
        i = i // le
        out.append(i)
        assert 0 <= i < self.n_rows
        return out

    def reset(self):
        P = []
        for s, p in self.isd:
            P.append(p)
        idx = np.random.choice(len(P), p=P)
        self.state = np.array([self.isd[idx][0]])
        return self.state

    def step(self, action):
        self.lastaction = action
        p = self.P_mat[np.asscalar(self.state), action]
        next_state = np.asscalar(np.random.choice(self.nb_states, 1, p=p))
        absorbing = next_state in self.absorbing_states
        self.reward = self.R_mat[self.state, action, next_state]
        self.state = np.array([next_state])
        return self.state, self.reward, absorbing, {}

    # def execute2(self, action):
    #     self.lastaction = action
    #     list_of_nextstates = self.P[self.state][action]
    #     next_state = np.asscalar(np.random.choice(list_of_nextstates[0], 1, p=list_of_nextstates[1]))
    #     self.reward = self.R_mat[self.state, action]
    #     self.state = next_state
    '''
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, goldlevel, enemypos, objectidx = self.decode(self.fromCompactToExtended[self.state])
        def ul(x): return "_" if x == " " else x
        if objectidx == 0:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'gray', highlight=True)
        elif objectidx == 1:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'blue', highlight=True)
        elif objectidx == 2:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'green', highlight=True)
        else: # both armor and key
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'red', highlight=True)
        di, dj = self.enemy_locs[enemypos]
        if (di,dj) != (taxirow, taxicol):
            out[1+di][2*dj+1] = utils.colorize('e', 'yellow')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({}) [gold: {}, object: {}]\n".format(self.action_names[self.lastaction], goldlevel, ['None', 'key', 'armor', 'key/armor'][objectidx]))
        else:
            outfile.write("  (-) [gold: {}, object: {}]\n".format(goldlevel,['None', 'key', 'armor', 'key/armor'][objectidx]))
    def show(self):
        dim = 200
        rows, cols = 4 + 0.5, 4
        if not hasattr(self, 'window'):
            root = Tk()
            self.window = gui.GUI(root)
            self.window.config(width=cols * (dim + 6), height=rows * (dim + 6))
            my_font = tkFont.Font(family="Arial", size=32, weight="bold")
            for r in range(4):
                for c in range(4):
                    x, y = 10 + c * (dim + 4), 10 + r * (dim + 4)
                    self.window.create_polygon([x, y, x + dim, y, x + dim, y + dim, x, y + dim], outline='black',
                                               fill='white', width=2)
            self.window.pack()
        self.window.update()
        self.window.mainloop()
    '''

    # def compute_matrix_form(self):
    #     nS = self.nb_states
    #     nA = self.max_nb_actions_per_state
    #     P_mat = np.zeros((nS,nA,nS), dtype=np.float)
    #     for s in range(nS):
    #         for a in range(nA):
    #             for next_state, p in zip(self.P[s][a][0], self.P[s][a][1]):
    #                 P_mat[s,a,next_state] += p
    #     return nS, nA, P_mat, self.R_mat

    def description(self):
        desc = {
            'name': type(self).__name__,
            'armor_move_prob': self.armor_move_prob,
            'armor_collect_prob': self.armor_collect_prob
        }
        return desc
