import cvxpy as cp
import numpy as np
from enum import Enum

class RewardSetType(object):
    FINITE = 'finite'
    BOX = 'Box'
    GENERAL_CONVEX = 'General-Convex-Body'


def solve_relaxed_characteristic_time_finite(mdp, policy, rewards)