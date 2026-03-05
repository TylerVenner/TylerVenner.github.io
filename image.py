import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

np.random.seed(1)

r = 0.6
n = 100

x = np.random.normal(0, 1, n)
y = r * x + np.random.normal(0, np.sqrt(1 - r**2), n)

mean_x, mean_y = np.mean(x), np.mean(y)
cov = np.cov(x, y)
eigvals, eigvecs = np.linalg.eigh(cov)

order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

chi2_val = chi2.ppf(0.95, df=2)
width = 2 * np.sqrt(eigvals[0] * chi2_val)
height = 2 * np.sqrt(eigvals[1] * chi2_val)

angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

beta = cov[0, 1] / cov[0, 0]

pc1 = eigvecs[:, 0]
pca_slope = pc1[1] / pc1[0]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#555555'
plt.rcParams['ytick.color'] = '#555555'

fig, ax = plt.subplots(figsize=(8, 8))

ax.grid(True, linestyle='--', alpha=0.4, color='#a3a3a3', zorder=0)

ax.scatter(x, y, alpha=0.65, color='#2563eb', edgecolor='white', linewidth=0.8, s=65, zorder=2)

ellipse = Ellipse((mean_x, mean_y),
                  width=width,
                  height=height,
                  angle=angle,
                  edgecolor='#475569',
                  facecolor='none',
                  lw=2,
                  linestyle='--',
                  zorder=3,
                  label='95% Data Ellipse')

ax.add_patch(ellipse)
x_vals = np.array([-3.5, 3.5])

ax.plot(x_vals,
        mean_y + pca_slope * (x_vals - mean_x),
        color='#334155', 
        lw=2.5,
        zorder=4,
        label='Major Axis (Orthogonal Fit)')

ax.plot(x_vals,
        mean_y + beta * (x_vals - mean_x),
        color='#dc2626', 
        lw=3,
        zorder=5,
        label=f'OLS Regression')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')

ax.axhline(0, color='#cccccc', linewidth=1, zorder=1)
ax.axvline(0, color='#cccccc', linewidth=1, zorder=1)

ax.set_xlabel('Standardized X', fontsize=11, fontweight='500', labelpad=10)
ax.set_ylabel('Standardized Y', fontsize=11, fontweight='500', labelpad=10)
ax.set_title(f'Geometry of Regression to the Mean (r = {r})', fontsize=14, fontweight='bold', pad=15)

ax.legend(loc='upper left', frameon=False, fontsize=10.5)

plt.tight_layout()

plt.savefig('regression_ellipse_transparent.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()