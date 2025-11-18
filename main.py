import numpy as np
import time
from Algorithm.revised_simplex import revised_simplex
from Algorithm.two_phase_revised_simplex import two_phase_simplex
from scipy.optimize import linprog
import pandas as pd

def compare_methods(A, b, c, problem_name, use_two_phase=True):

    result_dict = {"Problem": problem_name, "Variables": len(c), "Constraints": len(b)}
    
    start = time.perf_counter()
    try:
        if use_two_phase:
            x_our, z_our = two_phase_simplex(A, b, c)
        else:
            x_our, z_our = revised_simplex(A, b, c)
        time_our = time.perf_counter() - start
        iter_our = "N/A"  
        result_dict.update({"Our Obj": z_our, "Our Time": round(time_our, 8), "Our Iter": iter_our})
    except Exception as e:
        x_our, z_our, time_our = None, None, 0
        result_dict.update({"Our Obj": str(e), "Our Time": 0, "Our Iter": "N/A"})
    
  
    start = time.perf_counter()
    bounds = [(0, None) for _ in range(len(c))]
    try:
        result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
        time_scipy = time.perf_counter() - start
        if result.success:
            x_scipy, z_scipy, iter_scipy = result.x, result.fun, result.nit
        else:
            x_scipy, z_scipy, iter_scipy = None, None, "Fail"
        result_dict.update({"SciPy Obj": z_scipy, "SciPy Time": round(time_scipy, 8), "SciPy Iter": iter_scipy})
    except Exception as e:
        result_dict.update({"SciPy Obj": str(e), "SciPy Time": 0, "SciPy Iter": "Fail"})
    
    return result_dict


problems = [
    {"name": "Small Regular LP", "A": np.array([[1, 2, 1, 1],[2, 1, 2, 0]]),
     "b": np.array([4, 5]), "c": np.array([-3, -1, -3, 0]), "two_phase": False},

    {"name": "Medium LP", "A": np.array([[1, 1, 1, 1, 0],[2, 0, 1, 0, 1],[1, 2, 0, 1, 1]]),
     "b": np.array([10, 10, 12]), "c": np.array([-1, -2, -3, 0, -1]), "two_phase": True},
    

    {"name": "Large LP", "A": np.array([
        [1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 2]
    ]), "b": np.array([20, 18, 15, 25, 22, 19]), 
     "c": np.array([-3, -1, -4, -2, -1, -3, -2, -1, -3, -2, -1, -4]), "two_phase": True},
    
 
    {"name": "Infeasible LP", "A": np.array([[1, 1],[1, 1]]),
     "b": np.array([1, 3]), "c": np.array([1, 1]), "two_phase": True},

    {"name": "Unbounded LP", "A": np.array([[1, -1, 0],[-1, 1, 0]]),
     "b": np.array([1, -2]), "c": np.array([-1, -1, 0]), "two_phase": True},
 
    {"name": "Degenerate LP", "A": np.array([[1, 1, 0],[2, 2, 1]]),
     "b": np.array([2, 4]), "c": np.array([-1, -1, 0]), "two_phase": True},
]


results = []
for p in problems:
    res = compare_methods(p["A"], p["b"], p["c"], p["name"], p["two_phase"])
    results.append(res)

df = pd.DataFrame(results)
print("\nComparison Table:")
print(df[["Problem","Variables","Constraints","Our Obj","SciPy Obj","Our Time","SciPy Time"]])
