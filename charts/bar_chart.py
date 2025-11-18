import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("comparison_results.csv")
problems = df["Problem"]
our_times = df["Our Time"]
scipy_times = df["SciPy Time"]

plt.figure(figsize=(10,6))
bar_width = 0.35
index = range(len(problems))

plt.bar(index, our_times, bar_width, label="my Implementation")
plt.bar([i + bar_width for i in index], scipy_times, bar_width, label="SciPy linprog")

plt.xlabel("Problem")
plt.ylabel("Time (seconds)")
plt.title("Runtime Comparison")
plt.xticks([i + bar_width/2 for i in index], problems, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
