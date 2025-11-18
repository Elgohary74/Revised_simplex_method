import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


df = pd.read_csv("comparison_results.csv")
problems = df["Problem"]
our_times = df["Our Time"]
scipy_times = df["SciPy Time"]


objective_diff = np.abs(df["Our Obj"].replace("original problem is infeasible", 0).replace("Singular matrix", 0).astype(float) -
                        df["SciPy Obj"].fillna(0).astype(float))

plt.figure(figsize=(10,5))
plt.plot(problems, objective_diff, marker='o', linestyle='-', color='red')
plt.xlabel("Problem")
plt.ylabel("Objective Difference |Δz|")
plt.title("Objective Difference Between Our Implementation and SciPy")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


