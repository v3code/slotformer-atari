import random

import gym
import numpy as np
import skimage
from gym.utils import seeding
from gym import spaces
import sys
import copy

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

# import envs


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width / 2, r0 + width, r0 + width / 2], [c0 + width / 2, c0, c0 + width / 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width - 0.5, r0 + width - 0.5, r0], [c0, c0, c0 + width - 0.5, c0 + width - 0.5]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0 - 1.5, r0 + width - 1, r0 + width - 1], [c0 + width / 2 - 0.5, c0, c0 + width - 1]
    return skimage.draw.polygon(rr, cc, im_size)


def circle(r0, c0, width, im_size):
    radius = width / 2
    return skimage.draw.ellipse(r0 + radius - 0.5, c0 + radius - 0.5, radius, radius, im_size)


def cross(r0, c0, width, im_size):
    diff1 = width / 3
    diff2 = 2 * width / 3 - 1
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width - 0.5, r0 + width - 0.5,
          r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width - 0.5,
          c0 + width - 0.5, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)


def pentagon(r0, c0, width, im_size):
    diff1 = width / 3 - 1
    diff2 = 2 * width / 3 + 1
    rr = [r0 + width / 2, r0 + width, r0 + width, r0 + width / 2, r0]
    cc = [c0, c0 + diff1, c0 + diff2, c0 + width, c0 + width / 2]
    return skimage.draw.polygon(rr, cc, im_size)


def parallelogram(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width - 0.5, r0 + width - 0.5, r0], [c0, c0 + width / 2, c0 + width - 0.5, c0 + width - width / 2 - 0.5]
    return skimage.draw.polygon(rr, cc, im_size)


def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width / 2], [c0 + width - width / 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


class Shapes2d(gym.Env):
    GOAL = 'goal'
    STATIC_BOX = 'static_box'
    BOX = 'box'
    MOVED_BOXES_KEY = 'moved_boxes'
    MOVING_BOXES_KEY = 'moving_boxes'

    STEP_REWARD = -0.01
    OUT_OF_FIELD_REWARD = -0.1
    COLLISION_REWARD = -0.1
    DEATH_REWARD = -1
    HIT_GOAL_REWARD = 1
    DESTROY_GOAL_REWARD = -1
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n_boxes=5, n_static_boxes=0, n_goals=1, static_goals=True, width=5, embodied_agent=False,
                 return_state=False, observation_type='shapes', border_walls=True, channels_first=True,
                 channel_wise=False, seed=None, render_scale=10, ternary_interactions=False, do_reward_push_only=False):
        if n_static_boxes > 0:
            assert n_goals == 0 or static_goals, 'Cannot have movable goals with static objects.'

        if n_goals > 0 and not static_goals:
            assert n_static_boxes == 0, 'Cannot have static objects with movable goals'
        
        self.seed

        self.w = width
        self.n_boxes = n_boxes
        self.embodied_agent = embodied_agent
        self.ternary_interactions = ternary_interactions
        self.do_reward_push_only = do_reward_push_only

        self.goal_ids = set()
        self.static_box_ids = set()
        for k in range(self.n_boxes):
            box_id = self.n_boxes - k - 1
            if k < n_goals:
                self.goal_ids.add(box_id)
            elif k < n_goals + n_static_boxes:
                self.static_box_ids.add(box_id)
            else:
                break

        assert len(self.goal_ids) == n_goals
        assert len(self.static_box_ids) == n_static_boxes
        if self.embodied_agent:
            assert self.n_boxes > len(self.goal_ids) + len(self.static_box_ids)

        self.n_boxes_in_game = None
        self.static_goals = static_goals
        self.render_scale = render_scale
        self.border_walls = border_walls
        self.colors = get_colors(num_colors=max(9, self.n_boxes))
        self.observation_type = observation_type
        self.return_state = return_state
        self.channels_first = channels_first
        self.channel_wise = channel_wise

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }
        self.direction2action = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}

        self.np_random = None

        if self.embodied_agent:
            self.action_space = spaces.Discrete(4)
        else:
            n_movable_objects = self.n_boxes - len(self.goal_ids) * self.static_goals - len(self.static_box_ids)
            self.action_space = spaces.Discrete(4 * n_movable_objects)

        if self.observation_type in ('squares', 'shapes'):
            observation_shape = (self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels())
            if self.channels_first:
                observation_shape = (observation_shape[2], *observation_shape[:2])
            self.observation_space = spaces.Box(0, 255, observation_shape, dtype=np.uint8)
        else:
            raise ValueError(f'Invalid observation_type: {self.observation_type}.')

        self.state = None
        self.steps_taken = 0
        self.box_pos = np.zeros(shape=(self.n_boxes, 2), dtype=np.int32)
        self.speed = [{direction: 1 for direction in self.direction2action} for _ in range(self.n_boxes)]
        self.hit_goal_reward = [Shapes2d.HIT_GOAL_REWARD] * self.n_boxes
        self.render_mode = 'rgb_array'

        self.seed(seed)
        self.reset()

    def _get_image_channels(self):
        if not self.channel_wise:
            return 3

        return self.n_boxes

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        if self.observation_type == 'squares':
            image = self.render_squares()
        elif self.observation_type == 'shapes':
            image = self.render_shapes()
        else:
            assert False, f'Invalid observation type: {self.observation_type}.'

        if self.return_state:
            return np.array(self.state), image

        return image

    def render(self, mode=None):
        observation = self._get_observation()
        if self.return_state:
            return observation[1]

        return observation

    def reset(self):
        state = np.full(shape=[self.w, self.w], fill_value=-1, dtype=np.int32)

        # sample random locations for objects
        if self.embodied_agent:
            is_agent_in_main_area = self.np_random.random() > 4 * (self.w - 1) / self.w / self.w
            locs = self.np_random.choice((self.w - 2) ** 2, self.n_boxes - 1 + is_agent_in_main_area, replace=False)
            xs, ys = np.unravel_index(locs, [self.w - 2, self.w - 2])
            xs += 1
            ys += 1
            if not is_agent_in_main_area:
                agent_loc = self.np_random.choice(4 * (self.w - 1))
                side_id = agent_loc // (self.w - 1)
                if side_id == 0:
                    x = 0
                    y = agent_loc % (self.w - 1)
                elif side_id == 1:
                    x = agent_loc % (self.w - 1)
                    y = self.w - 1
                elif side_id == 2:
                    x = self.w - 1
                    y = self.w - 1 - agent_loc % (self.w - 1)
                elif side_id == 3:
                    x = self.w - 1 - agent_loc % (self.w - 1)
                    y = 0
                else:
                    raise ValueError(f'Unexpected side_id={side_id}')

                xs = np.append(x, xs)
                ys = np.append(y, ys)
        else:
            locs = self.np_random.choice(self.w ** 2, self.n_boxes, replace=False)
            xs, ys = np.unravel_index(locs, [self.w, self.w])

        # populate state with locations
        for i, (x, y) in enumerate(zip(xs, ys)):
            state[x, y] = i
            self.box_pos[i, :] = x, y

        self.state = state
        self.steps_taken = 0
        self.n_boxes_in_game = self.n_boxes - len(self.goal_ids) - int(self.embodied_agent) - int(self.do_reward_push_only)
        if not self.embodied_agent:
            self.n_boxes_in_game -= len(self.static_box_ids) * int(self.static_goals)

        info = {Shapes2d.MOVING_BOXES_KEY: self._get_all_moving_boxes()}
        return self._get_observation()

    def _get_type(self, box_id):
        if box_id in self.goal_ids:
            return Shapes2d.GOAL
        elif box_id in self.static_box_ids:
            return Shapes2d.STATIC_BOX
        else:
            return Shapes2d.BOX

    def _destroy_box(self, box_id):
        box_pos = self.box_pos[box_id]
        self.state[box_pos[0], box_pos[1]] = -1
        self.box_pos[box_id] = -1, -1
        if self._get_type(box_id) == Shapes2d.BOX or (
                not self.embodied_agent and self._get_type(box_id) == Shapes2d.STATIC_BOX):
            self.n_boxes_in_game -= 1

    def _move(self, box_id, new_pos):
        old_pos = self.box_pos[box_id]
        self.state[old_pos[0], old_pos[1]] = -1
        self.state[new_pos[0], new_pos[1]] = box_id
        self.box_pos[box_id] = new_pos

    def _is_free_cell(self, pos):
        return self.state[pos[0], pos[1]] == -1

    def _get_occupied_box_id(self, pos):
        return self.state[pos[0], pos[1]]

    def _make_step(self, action, simulate=False, return_observation=True, increment_step=True):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        vec = self.directions[action % len(self.directions)]
        box_id = action // len(self.directions)
        box_old_pos = self.box_pos[box_id]
        box_new_pos = box_old_pos + vec
        box_type = self._get_type(box_id)

        done = False
        reward = Shapes2d.STEP_REWARD if increment_step else 0
        moved_boxes = [0] * self.n_boxes
        moved_boxes[box_id] = 1

        if not self._is_in_grid(box_old_pos, box_id):
            # This box is out of the game. There is nothing to do.
            pass
        elif not self._is_in_grid(box_new_pos, box_id):
            reward += Shapes2d.OUT_OF_FIELD_REWARD
            if not self.border_walls:
                # push out of grid, destroy object or finish episode if an agent is out of the grid
                if box_type == Shapes2d.GOAL:
                    reward += Shapes2d.DESTROY_GOAL_REWARD
                elif self.embodied_agent:
                    reward += Shapes2d.DEATH_REWARD
                    done = True

                if not simulate:
                    self._destroy_box(box_id)
        elif not self._is_free_cell(box_new_pos):
            # push into another box
            another_box_id = self._get_occupied_box_id(box_new_pos)
            another_box_type = self._get_type(another_box_id)

            if box_type == Shapes2d.BOX:
                if another_box_type == Shapes2d.BOX:
                    if self.ternary_interactions:
                        moved_boxes[another_box_id] = 1
                        another_box_new_pos = box_new_pos + vec
                        if self._is_in_grid(another_box_new_pos, another_box_id):
                            if self._is_free_cell(another_box_new_pos):
                                if not simulate:
                                    self._move(another_box_id, another_box_new_pos)
                                    self._move(box_id, box_new_pos)
                            elif self._get_type(self._get_occupied_box_id(another_box_new_pos)) == Shapes2d.GOAL:
                                reward += self.hit_goal_reward[another_box_id]
                                if not simulate:
                                    self._destroy_box(another_box_id)
                                    self._move(box_id, box_new_pos)
                            else:
                                reward += Shapes2d.COLLISION_REWARD
                        else:
                            reward += Shapes2d.OUT_OF_FIELD_REWARD
                            if not self.border_walls:
                                if not simulate:
                                    self._destroy_box(another_box_id)
                                    self._move(box_id, box_new_pos)
                    else:
                        reward += Shapes2d.COLLISION_REWARD
                elif another_box_type == Shapes2d.GOAL:
                    if self.embodied_agent or self.do_reward_push_only:
                        reward += Shapes2d.COLLISION_REWARD
                    else:
                        reward += self.hit_goal_reward[box_id]
                        if not simulate:
                            self._destroy_box(box_id)
                else:
                    assert another_box_type == Shapes2d.STATIC_BOX
                    reward += Shapes2d.COLLISION_REWARD
            elif box_type == Shapes2d.GOAL:
                if another_box_type in (Shapes2d.BOX, Shapes2d.STATIC_BOX):
                    if not simulate:
                        self._destroy_box(another_box_id)
                        self._move(box_id, box_new_pos)
                    reward += self.hit_goal_reward[another_box_id]
                else:
                    assert another_box_type == Shapes2d.GOAL
                    reward += Shapes2d.COLLISION_REWARD
            else:
                assert False, f'Cannot move a box of type:{box_type}'
        else:
            # pushed into open space, move box
            if not simulate:
                self._move(box_id, box_new_pos)

        info = {Shapes2d.MOVED_BOXES_KEY: moved_boxes}

        if increment_step:
            self.steps_taken += 1

        if self.n_boxes_in_game == 0:
            done = True
            info['is_success'] = True

        observation = self._get_observation() if return_observation else None

        return observation, reward, done, done, info

    def _get_all_moving_boxes(self):
        moving_boxes = []
        for action in range(self.action_space.n):
            _, _, _, _, info = self._make_step(action, simulate=True, return_observation=False, increment_step=False)
            moving_boxes.append(info[Shapes2d.MOVED_BOXES_KEY])

        return moving_boxes

    def step(self, action):
        box_id = action // len(self.directions)
        direction = tuple(self.directions[action % len(self.directions)].tolist())
        speed = self.speed[box_id][direction]
        reward = 0
        moving_boxes = None
        moved_boxes = None

        for sub_step in range(speed):
            is_last_sub_step = sub_step == speed - 1
            observation, sub_reward, done, info = self._sub_step(action, return_observation=is_last_sub_step,
                                                                 increment_step=is_last_sub_step)
            reward += sub_reward
            if moved_boxes is None:
                moved_boxes = info[Shapes2d.MOVED_BOXES_KEY]
            else:
                moved_boxes = [max(flag1, flag2) for flag1, flag2 in zip(moved_boxes, info[Shapes2d.MOVED_BOXES_KEY])]

            if moving_boxes is None:
                moving_boxes = info[Shapes2d.MOVING_BOXES_KEY]
            else:
                for i, new_flags in enumerate(info[Shapes2d.MOVING_BOXES_KEY]):
                    moving_boxes[i] = [max(old_flag, new_flag) for old_flag, new_flag in
                                       zip(moving_boxes[i], new_flags)]

        info[Shapes2d.MOVING_BOXES_KEY] = moving_boxes
        info[Shapes2d.MOVED_BOXES_KEY] = moved_boxes

        return observation, reward, done, done, info

    def _sub_step(self, action, return_observation, increment_step):
        observation, reward, done, _, info = self._make_step(action, simulate=False, return_observation=return_observation,
                                                          increment_step=increment_step)
        info[Shapes2d.MOVING_BOXES_KEY] = self._get_all_moving_boxes()

        return observation, reward, done, info

    def _is_in_grid(self, point, box_id):
        if not self.embodied_agent or box_id == 0:
            return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

        return (1 <= point[0] < self.w - 1) and (1 <= point[1] < self.w - 1)

    def print(self, message=''):
        state = self.state
        chars = {-1: '.'}
        for box_id in range(self.n_boxes):
            if box_id in self.goal_ids:
                chars[box_id] = 'x'
            elif box_id in self.static_box_ids:
                chars[box_id] = '#'
            else:
                chars[box_id] = '@'

        pretty = "\n".join(["".join([chars[x] for x in row]) for row in state])
        print(pretty)
        print(message)

    def clone_full_state(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def restore_full_state(self, state_dict):
        self.__dict__.update(state_dict)

    def get_action_meanings(self):
        return ["down", "left", "up", "right"] * (
                self.n_boxes - len(self.static_box_ids) - len(self.goal_ids) * self.static_goals)

    def render_squares(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels()),
                      dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            if self.channel_wise:
                im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255

        return im.astype(dtype=np.uint8)

    def render_shapes(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels()),
                      dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            shape_id = idx % 8
            if shape_id == 0:
                rr, cc = circle(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 1:
                rr, cc = triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 2:
                rr, cc = square(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 3:
                rr, cc = parallelogram(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 4:
                rr, cc = cross(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 5:
                rr, cc = diamond(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 6:
                rr, cc = pentagon(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            else:
                rr, cc = scalene_triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)

            if self.channel_wise:
                im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255
        return im.astype(np.uint8)