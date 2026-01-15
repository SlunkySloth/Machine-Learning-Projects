import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('tn_election_data.csv')

# Separate data by alliance
dmk_data = data[data['Alliance'] == 'DMK+']
aiadmk_data = data[data['Alliance'] == 'AIADMK+']

# Prepare X (Year) and y (VoteShare)
X_dmk = dmk_data[['Year']].values
y_dmk = dmk_data['VoteShare'].values

X_aiadmk = aiadmk_data[['Year']].values
y_aiadmk = aiadmk_data['VoteShare'].values

# Standard Regression for Trend
degree = 2
poly = PolynomialFeatures(degree=degree)

# Train DMK
X_dmk_poly = poly.fit_transform(X_dmk)
model_dmk = LinearRegression()
model_dmk.fit(X_dmk_poly, y_dmk)

# Train AIADMK
X_aiadmk_poly = poly.fit_transform(X_aiadmk)
model_aiadmk = LinearRegression()
model_aiadmk.fit(X_aiadmk_poly, y_aiadmk)

# Raw Prediction for 2026
year_2026 = np.array([[2026]])
year_2026_poly = poly.fit_transform(year_2026)

raw_dmk = model_dmk.predict(year_2026_poly)[0]
raw_aiadmk = model_aiadmk.predict(year_2026_poly)[0]

# --- TVK INTEGRATION & NORMALIZATION ---
tvk_share = 25.0
others_estimate = 10.0
available_share = 100.0 - tvk_share - others_estimate

total_raw = raw_dmk + raw_aiadmk
scaling_factor = available_share / total_raw

final_dmk = raw_dmk * scaling_factor
final_aiadmk = raw_aiadmk * scaling_factor

print(f"DMK+: {final_dmk:.2f}%")
print(f"AIADMK+: {final_aiadmk:.2f}%")
print(f"TVK: {tvk_share:.2f}%")
print(f"Others: {others_estimate:.2f}%")

# --- IMPROVED VISUALIZATION ---
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2)

# 1. Stacked Area Chart (Historical Trend)
ax1 = fig.add_subplot(gs[0, :])

# Prepare full timeline data for plotting area
years_hist = dmk_data['Year'].values
dmk_hist = dmk_data['VoteShare'].values
aiadmk_hist = aiadmk_data['VoteShare'].values

# For "Others" in history (approximate as 100 - DMK - AIADMK)
others_hist = 100 - dmk_hist - aiadmk_hist

# Append 2026 predictions
years_all = np.append(years_hist, 2026)
dmk_all = np.append(dmk_hist, final_dmk)
aiadmk_all = np.append(aiadmk_hist, final_aiadmk)
tvk_all = np.append(np.zeros_like(years_hist), tvk_share) # TVK was 0 before 2026
others_all = np.append(others_hist, others_estimate)

# Stackplot
ax1.stackplot(years_all, dmk_all, aiadmk_all, tvk_all, others_all, 
              labels=['DMK+', 'AIADMK+', 'TVK', 'Others'],
              colors=['#ff4d4d', '#4dff4d', '#ffab00', '#cccccc'], alpha=0.8)

ax1.set_title('Tamil Nadu Electoral Landscape Evolution (1991-2026)')
ax1.set_ylabel('Vote Share (%)')
ax1.set_xlim(1991, 2026)
ax1.set_ylim(0, 100)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Donut Chart (2026 Snapshot)
ax2 = fig.add_subplot(gs[1, 0])
sizes = [final_dmk, final_aiadmk, tvk_share, others_estimate]
labels = ['DMK+', 'AIADMK+', 'TVK', 'Others']
colors = ['#ff4d4d', '#4dff4d', '#ffab00', '#cccccc']
explode = (0.05, 0, 0, 0) # explode the winner slightly (assuming DMK wins) if needed, or just flat

ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
# Draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax2.add_artist(centre_circle)
ax2.set_title('2026 Projected Vote Share')

# 3. Bar Chart (Comparison)
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(labels, sizes, color=colors)
ax3.set_title('2026 Comparison')
ax3.set_ylabel('Vote Share (%)')
for i, v in enumerate(sizes):
    ax3.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig('election_dashboard.png')
print("Dashboard saved as election_dashboard.png")
