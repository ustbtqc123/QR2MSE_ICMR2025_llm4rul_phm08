import matplotlib.pyplot as plt
import numpy as np

# Data construction
x1 = np.linspace(0, 25, 50)  # Blue dashed line (RUL change)
y1 = 150 - (150 - 125) * (x1 / 25)  # Simple interpolation

x2 = np.linspace(0, 25, 50)  # Red horizontal line (Piecewise model)
y2 = np.ones_like(x2) * 125

x3 = np.linspace(25, 150, 200)  # Red decreasing line
y3 = 125 - (125 / (150 - 25)) * (x3 - 25)

plt.figure(figsize=(8,5))
plt.plot(x1, y1, 'b--', label='Actual RUL Change')
plt.plot(x2, y2, 'r-', label='Piecewise Fitted RUL Model')
plt.plot(x3, y3, 'r-')

plt.xlabel('Operating Cycle', fontsize=16)
plt.ylabel('RUL', fontsize=16)
plt.xlim(0, 160)
plt.ylim(0, 160)
plt.grid(False)

# 设置图例字体大小
plt.legend(fontsize=14)

# 设置刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=14)

# 去除边框空白
plt.tight_layout()

# 保存图片
plt.savefig("RUL_target_function.png", bbox_inches='tight', dpi=300)
