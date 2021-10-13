import math
import random
from typing import List, Union

import matplotlib.pyplot as plt
import mlrose
import numpy as np
import pypot.vrep
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from poppy_ergo_jr import PoppyErgoJr
from pypot.dynamixel import DxlXL320Motor

num_flies = 10
default_bounds = (1, 100)
goto_time = 10


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def get_random_pos(bounds: Union[List[tuple], tuple] = default_bounds) -> tuple:
    if len(bounds) == 0:
        bounds = default_bounds
    if isinstance(bounds, list):
        if len(bounds) < 3:
            result = [0, 0, 0]
            for i in range(0, len(result)):
                try:
                    item = random.randint(bounds[i][0], bounds[i][1])
                except IndexError:
                    item = random.randint(default_bounds[0], default_bounds[1])
                result[i] = item
            return tuple(result)
        else:
            pos_x = random.randint(bounds[0][0], bounds[0][1])
            pos_y = random.randint(bounds[1][0], bounds[1][1])
            pos_z = random.randint(bounds[2][0], bounds[2][1])
            return pos_x, pos_y, pos_z
    else:
        pos_x = random.randint(default_bounds[0], default_bounds[1])
        pos_y = random.randint(default_bounds[0], default_bounds[1])
        pos_z = random.randint(default_bounds[0], default_bounds[1])
        return pos_x, pos_y, pos_z


def ik_goto(poppy, x, y, z):
    destination_pos = np.asarray([float(x), float(y), float(z)])

    poppy.chain.goto(destination_pos, float(goto_time), True, True)


def create_flies(count):
    flies = []
    for _ in range(0, count):
        flies.append(np.asarray(get_random_pos()))
    return flies


def calculate_path(flies):
    print(f"current distance : {calculate_total_len(flies)}")

    reordered_flies = create_distance_array(flies)

    fitness_dists = mlrose.TravellingSales(distances=reordered_flies)
    problem_fit = mlrose.TSPOpt(length=10, fitness_fn=fitness_dists, maximize=False)
    res = mlrose.algorithms.hill_climb(problem_fit, 10_000)

    print(f"new distance : {calculate_total_len(flies, order=res[0])}")
    show_flies(flies, order=res[0])
    return res[0]


def calculate_distance(pos1, pos2):
    d = math.sqrt(math.pow(pos2[0] - pos1[0], 2) + math.pow(pos2[1] - pos1[1], 2) + math.pow(pos2[2] - pos1[2], 2))
    return d


def create_distance_array(flies):
    distance_array = []
    for i in range(0, len(flies)):
        for j in range(0, len(flies)):
            pos_i = flies[i]
            pos_j = flies[j]
            if not (pos_i == pos_j).all():
                distance = calculate_distance(pos_i, pos_j)
                distance_array.append((i, j, distance))
    return distance_array


def calculate_total_len(flies, order=None):
    total = 0
    if order is not None:
        for i, pos in enumerate(order):
            pos1 = flies[pos]
            if (len(order) - 1) <= i + 1:
                return total
            j = order[i + 1]
            pos2 = flies[j]
            total += calculate_distance(pos1, pos2)
    else:
        for i in range(0, len(flies)):
            pos1 = flies[i]
            if (len(flies) - 1) <= i + 1:
                return total
            pos2 = flies[i + 1]
            total += calculate_distance(pos1, pos2)
    return total


def show_flies(flies, order=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    flies = np.asarray(flies)
    xs = flies[:, 0].tolist()
    ys = flies[:, 1].tolist()
    zs = flies[:, 2].tolist()

    ax.scatter(xs, ys, zs)

    for i in range(0, len(flies)):
        ax.text(flies[i][0], flies[i][1], flies[i][2], str(i), size=10, zorder=1,
                color='k')
    if order is None:
        for i in range(0, len(flies) - 1):
            next_flies = flies[i + 1]
            arw = Arrow3D([flies[i][0], next_flies[0]], [flies[i][1], next_flies[1]], [flies[i][2], next_flies[2]],
                          arrowstyle="->", color="red", lw=1,
                          mutation_scale=10)
            ax.add_artist(arw)

    else:
        for i, pos in enumerate(order):
            next_pos = order[i + 1]
            next_flies = flies[next_pos]
            arw = Arrow3D([flies[pos][0], next_flies[0]], [flies[pos][1], next_flies[1]],
                          [flies[pos][2], next_flies[2]],
                          arrowstyle="->", color="red", lw=1,
                          mutation_scale=10)
            ax.add_artist(arw)
            if (len(order) - 1) <= i + 1:
                break

    ax.set_xlabel('X', fontsize=10, labelpad=10)
    ax.set_ylabel('Y', fontsize=10, labelpad=10)
    ax.set_zlabel('Z', fontsize=10, labelpad=10)

    fig.show()


if __name__ == '__main__':
    poppy = PoppyErgoJr(simulator="vrep", scene="poppy_ergo_jr_holder.ttt", camera="dummy")
    claw_motor: DxlXL320Motor = poppy.m6

    flies = create_flies(num_flies)
    show_flies(flies)
    path = calculate_path(flies)

    for pos in path:
        claw_motor.goto_position(20, 5, wait=True)
        ik_goto(poppy, *flies[pos])
        claw_motor.goto_position(-20, 5, wait=True)

    poppy.stop_simulation()
    pypot.vrep.close_all_connections()
