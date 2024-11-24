import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from typing import Optional, Tuple, List, Callable
from scipy.linalg._fblas import dger, dgemm
from typing import NamedTuple
from itertools import product
from multireward_ope.tabular.policy import Policy, PolicyFactory


def find_optimal_actions(Q: np.ndarray, tol: float = 1e-3) -> List[np.ndarray]:
    num_states = Q.shape[0]
    optimal_actions_per_state = []

    for s in range(num_states):
        max_actions = np.argwhere(np.isclose(Q[s],np.max(Q[s]))).flatten()
        optimal_actions_per_state.append(max_actions)

    return optimal_actions_per_state

def generate_optimal_policies(Q: np.ndarray):
    optimal_actions_per_state = find_optimal_actions(Q)
    all_possible_policies = list(product(*optimal_actions_per_state))

    # Convert to NumPy array
    policies_array = np.array(all_possible_policies, dtype=np.ulong)
    return policies_array

def policy_evaluation(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        policy: Policy,
        V0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6,
        max_iter: int  = 1000) -> NDArray[np.float64]:
    """Policy evaluation

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        policy (Optional[NDArray[np.ulong]], optional): policy
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Value function
    """
    
    NS, NA = P.shape[:2]
    # Initialize values
    if V0 is None:
        V0 = np.zeros(NS)
    
    V = V0.copy()
    iter = 0
    for iter in range(max_iter):
        Delta = 0
        V_next = np.array([P[s, policy[s]] @ (R[s, policy[s]] + gamma * V) for s in range(NS)])
        
        Delta = np.max([Delta, np.abs(V_next - V).max()])
        V = V_next

        if Delta < atol:
            break
    return V
        

def policy_iteration(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi0: Optional[Policy] = None,
        V0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6,
        max_iter: int = 1000) -> Tuple[NDArray[np.float64], Policy, NDArray[np.float64]]:
    """Policy iteration

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi0 (Optional[Policy], optional): Initial policy. Defaults to None.
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Optimal value function
        Policy: Optimal policy
        NDArray[np.float64]: Optimal Q function
    """
    
    NS, NA = P.shape[:2]

    # Initialize values    
    V = V0 if V0 is not None else np.zeros(NS)
    pi = pi0 if pi0 is not None else PolicyFactory.random(NS, NA)
    next_pi = np.zeros_like(pi)
    policy_stable = False

    iter = 0
    while not policy_stable and iter < max_iter:
        policy_stable = True
        #V = policy_evaluation(gamma, P, R, pi[0], V, atol, max_iter)
        V = policy_evaluation(gamma, P, R[..., np.newaxis] if len(R.shape) == 2 else R, pi[0], atol, max_iter)
        Q = np.array([[P[s,a] @ (R[s,a] + gamma * V) for a in range(NA)] for s in range(NS)])
        next_pi = generate_optimal_policies(Q)
        
        if np.any(next_pi.shape != pi.shape) or np.any(next_pi != pi):
            policy_stable = False
        pi = next_pi
        iter += 1

    return V, pi, Q

def project_omega(
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        force_policy: bool = True) -> NDArray[np.float64]:
    """Project omega using navigation constraints

    Parameters
    ----------
    x : NDArray[np.float64]
        Allocation vector to project
    P : NDArray[np.float64]
        Transition matrix (S,A,S)

    Returns
    -------
    NDArray[np.float64]
        The projected allocation vector
    """
    ns, na = P.shape[:2]
    omega = cp.Variable((ns, na), nonneg=True)
    constraints = [cp.sum(omega) == 1]
    constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])
    if force_policy:
        constraints.append(omega >= 1e-15)
        constraints.extend(
            [omega[s,a] == x[s,a]/np.sum(x[s]) * cp.sum(omega[s]) for a in range(na) for s in range(ns)])
    else:
        constraints.append(omega >= 1e-4)
  
    objective = cp.Minimize(0.5 * cp.norm(x - omega, 2)**2)
    problem = cp.Problem(objective, constraints)
        
    res = problem.solve(verbose=False, solver=cp.MOSEK)
    #print(f'res: {res}')
    return omega.value

def compute_stationary_distribution(
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        solver: str) -> NDArray[np.float64]:
    """Project omega using navigation constraints

    Parameters
    ----------
    x : NDArray[np.float64]
        Allocation vector to project
    P : NDArray[np.float64]
        Transition matrix (S,A,S)

    Returns
    -------
    NDArray[np.float64]
        The projected allocation vector
    """
    ns, na = P.shape[:2]
    omega = cp.Variable((ns, na), nonneg=True)
    constraints = [cp.sum(omega) == 1, omega >= 1e-15]
    constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])
    constraints.extend([omega[s,a] == x[s,a]/np.sum(x[s]) * cp.sum(omega[s]) for a in range(na) for s in range(ns)])
  
    problem = cp.Problem(cp.Minimize(1), constraints)
    res = problem.solve(verbose=False, solver=solver)
    return omega.value


def is_positive_definite(x: np.ndarray, atol: float = 1e-9) -> bool:
    """Check if a matrix is positive definite
    Args:
        x (np.ndarray): matrix
        atol (float, optional): absolute tolerance. Defaults to 1e-9.
    Returns:
        bool: Returns True if the matrix is positive definite
    """    
    return np.all(np.linalg.eigvals(x) > atol)

def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 0) -> bool:
    """Check if a matrix is symmetric
    Args:
        a (np.ndarray): matrix to check
        rtol (float, optional): relative tolerance. Defaults to 1e-05.
        atol (float, optional): absolute tolerance. Defaults to 1e-08.
    Returns:
        bool: returns True if the matrix is symmetric
    """    
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def mean_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mean and covariance of a matrix
    See https://groups.google.com/g/scipy-user/c/FpOU4pY8W2Y
    Args:
        X (np.ndarray): _description_
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Mean,Covariance) tuple
    """   
    n, p = X.shape
    m = X.mean(axis=0)
    # covariance matrix with correction for rounding error
    # S = (cx'*cx - (scx'*scx/n))/(n-1)
    # Am Stat 1983, vol 37: 242-247.
    cx = X - m
    scx = cx.sum(axis=0)
    scx_op = dger(-1.0/n,scx,scx)
    S = dgemm(1.0, cx.T, cx.T, beta=1.0,
    c = scx_op, trans_a=0, trans_b=1, overwrite_c=1)
    S[:] *= 1.0/(n-1)
    return m, S.T

def unit_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b / np.dot(b,b)  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w) #/np.linalg.norm(w))
    return np.array(basis)

def find_interior(halfspaces: NDArray[np.float64], solver: str = cp.CLARABEL) -> NDArray[np.float64]:
    """Returns a point clearly inside the region defined by halfspaces.

    Args:
        halfspaces (NDArray[np.float64]): Stacked Inequalities of the form Ax + b <= 0 in
                                           format [A; b]

    Returns:
        NDArray[np.float64]: A feasible point
    """
    A = halfspaces[:,:-1]
    b = halfspaces[:,-1]
    x = cp.Variable(A.shape[1])
    y = cp.Variable()

    constraints = [
        A @ x + y * np.linalg.norm(A[i], ord=2) <= -b for i in range(A.shape[0])
    ]

    obj = cp.Maximize(y)
    problem = cp.Problem(obj, constraints)
    res = problem.solve(solver=solver)
    return x.value
