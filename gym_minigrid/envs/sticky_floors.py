from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np
from gym import spaces


def manhattan_dist(a, b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])


class StickyFloorEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=[(1, 1), (8-2, 8-2)],
        agent_start_dirs=[0, 0],
        goals = [(8-2, 8-2), (1, 1)],
        min_prob=0.01,
        reward_type='manhattan_dist',
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dirs = agent_start_dirs
        self.goals = goals
        self._min_prob = min_prob
        self._num_grid_configs = len(goals)
        self._grid_config_index = 0
        self._reward_type = reward_type

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=0,
                high=32,
                shape=(2,),
                dtype='float32'
            ),
            'direction': spaces.Box(
                low=0,
                high=4,
                shape=(1,),
                dtype='float32'
            ),
            'goal_pos': spaces.Box(
                low=0,
                high=32,
                shape=(2,),
                dtype='float32'
            ),
        })


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        middle_x = (width - 1) / 2
        middle_y = (height - 1) / 2
        for x in range(1, width-1):
            for y in range(1, height-1):
                norm_delta_middle_x = (x - middle_x) / (width - 2 - middle_x)
                norm_delta_middle_y = (y - middle_y) / (height - 2 - middle_y)
                sticky_prob = self._sticky_prob(norm_delta_middle_x, norm_delta_middle_y)
                self.grid.sticky_floor(x, y, sticky_prob)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self._grid_config_index = (self._grid_config_index + 1) % self._num_grid_configs
        self.grid.set(*self.goals[self._grid_config_index], Goal())
        self.agent_pos = self.agent_start_pos[self._grid_config_index]
        self.agent_dir = self.agent_start_dirs[self._grid_config_index]

        self.mission = "get to the green goal square"

    def _sticky_prob(self, norm_delta_middle_x, norm_delta_middle_y):
        return np.clip(np.abs((norm_delta_middle_x - norm_delta_middle_y) / 2), 0, 1 - self._min_prob)

    def gen_obs(self):
        """
        Generate the agent's view
        """

        obs = {
            'agent_pos': np.array(self.agent_pos),
            'direction': np.array([self.agent_dir]),
            'goal_pos': np.array(self.goals[self._grid_config_index]),
            # 'mission': self.mission
        }
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self._reward_type == "manhattan_dist":
            goal = self.goals[self._grid_config_index]
            reward = -manhattan_dist(self.agent_pos, goal)
        return obs, reward, done, {}


class UnStickyFloorEnv8x8(StickyFloorEnv):
    def __init__(self, **kwargs):
        super().__init__(
            size=8,
            agent_start_pos=[(1, 1)],#, (8-2, 8-2)],
            agent_start_dirs=[0],#, 0],
            goals = [(8-2, 8-2)],#, (1, 1)],
            **kwargs)

    def _sticky_prob(self, norm_delta_middle_x, norm_delta_middle_y):
        return 0


class StickyFloorEnv8x8(StickyFloorEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8,
                         agent_start_pos=[(1, 1), (8-2, 8-2)],
                         agent_start_dirs=[0, 0],
                         goals = [(8-2, 8-2), (1, 1)],
                         **kwargs)

class StickyFloorEnv16x16(StickyFloorEnv):
    def __init__(self, **kwargs):
        super().__init__(
            size=16,
            agent_start_pos=[(1, 1), (16-2, 16-2)],
            agent_start_dirs=[0, 0],
            goals = [(16-2, 16-2), (1, 1)],
            **kwargs
        )

class StickyFloorExpGradEnv(StickyFloorEnv):
    def _sticky_prob(self, norm_delta_middle_x, norm_delta_middle_y):
        norm_dist_from_diagonal = np.clip(np.abs((norm_delta_middle_x - norm_delta_middle_y) / 2), 0, 0.999)
        a = 0.5
        return np.clip(np.exp(-a / norm_dist_from_diagonal), 0, 1 - self._min_prob)

class StickyFloorExpGradEnv16x16(StickyFloorExpGradEnv):
    def __init__(self):
        super().__init__(size=16)


register(
    id='MiniGrid-StickyFloor-8x8-v0',
    entry_point='gym_minigrid.envs:StickyFloorEnv8x8'
)

register(
    id='MiniGrid-UnStickyFloor-8x8-v0',
    entry_point='gym_minigrid.envs:UnStickyFloorEnv8x8'
)

register(
    id='MiniGrid-StickyFloor-16x16-v0',
    entry_point='gym_minigrid.envs:StickyFloorEnv16x16'
)

register(
    id='MiniGrid-StickyFloorExpGrad-16x16-v0',
    entry_point='gym_minigrid.envs:StickyFloorExpGradEnv16x16'
)
