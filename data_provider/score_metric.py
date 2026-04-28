import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
x = np.linspace(-50, 50, N)
y = []
# Calculate score
for i in range(N):
    if x[i] < 0:
        score = np.exp(-x[i] / 13) - 1
    else:
        score = np.exp(x[i] / 10) - 1
    y.append(score)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2)

# Vertical lines
plt.axvline(x=-10, color='k', linestyle='--', linewidth=1.2)
plt.axvline(x=10, color='k', linestyle='--', linewidth=1.2)

# Text annotations (增大字号)
plt.text(-20, 110, 'Early Prediction', ha='center', fontsize=14)
plt.text(0, 110, 'Timely Prediction', ha='center', fontsize=14)
plt.text(20, 110, 'Late Prediction', ha='center', fontsize=14)

# Labels with larger font
plt.xlabel('Prediction Error', fontsize=18)
plt.ylabel('Score', fontsize=18)

# Tick font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Axis limits
plt.xlim(-50, 50)
plt.ylim(0, 150)

# Grid style
plt.grid(True, linestyle='--', alpha=0.6)

# Tight layout and save
plt.tight_layout()
plt.savefig("Score_function.png", bbox_inches='tight', dpi=300)
# plt.show()
