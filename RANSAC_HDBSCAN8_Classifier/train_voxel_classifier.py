import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
df = pd.read_csv('./gt_runs/gt_bboxes_run_05_merge_clean.csv')
print(f"Total bounding boxes: {len(df)}")
print(f"\nClass distribution:")
print(df['class_label'].value_counts().to_string())
print()

# ── features from bbox geometry ──
df['height'] = df['bbox_height']
df['width'] = df['bbox_width']
df['length'] = df['bbox_length']

# Aspect ratios
df['height_to_width'] = df['height'] / (df['width'] + 1e-6)
df['height_to_length'] = df['height'] / (df['length'] + 1e-6)
df['width_to_length'] = df['width'] / (df['length'] + 1e-6)

# Volume and footprint
df['volume'] = df['width'] * df['length'] * df['height']
df['footprint'] = df['width'] * df['length']
df['max_horizontal'] = df[['width', 'length']].max(axis=1)
df['min_horizontal'] = df[['width', 'length']].min(axis=1)
df['horizontal_aspect'] = df['max_horizontal'] / (df['min_horizontal'] + 1e-6)

# Elongation (how "cable-like" is it?)
df['elongation'] = df['max_horizontal'] / (df['height'] + 1e-6)

# Verticality (how "pole-like" is it?)
df['verticality'] = df['height'] / (df['max_horizontal'] + 1e-6)

# Z-center (how high is the object?)
df['z_center'] = df['bbox_center_z']

# Point density
df['point_density'] = df['num_points'] / (df['volume'] + 1e-6)

# ── Features for classification ──
feature_cols = [
    'height', 'width', 'length',
    'height_to_width', 'height_to_length', 'width_to_length',
    'volume', 'footprint', 'max_horizontal', 'min_horizontal',
    'horizontal_aspect', 'elongation', 'verticality',
    'z_center', 'num_points', 'point_density'
]

# ══════════════════════════════════════════════════════════════
# PART 1: DESCRIPTIVE STATISTICS PER CLASS
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: DESCRIPTIVE STATS PER CLASS (median values)")
print("=" * 70)

key_features = ['height', 'width', 'length', 'height_to_width',
                'elongation', 'verticality', 'volume', 'num_points',
                'horizontal_aspect', 'point_density']

stats = df.groupby('class_label')[key_features].agg(['median', 'std'])
for feat in key_features:
    print(f"\n  {feat}:")
    for cls in ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']:
        if cls in stats.index:
            med = stats.loc[cls, (feat, 'median')]
            std = stats.loc[cls, (feat, 'std')]
            print(f"    {cls:20s}: median={med:10.2f}  std={std:10.2f}")

# ══════════════════════════════════════════════════════════════
# PART 2: RANDOM FOREST CLASSIFICATION (cross-validated)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: RANDOM FOREST CROSS-VALIDATION")
print("=" * 70)

X = df[feature_cols].values
y = df['class_label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified 5-fold CV
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
print(f"\n  5-Fold CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")

# Also try F1
f1_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='f1_weighted')
print(f"  5-Fold CV F1 (weighted): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# Per-class report on one full fit
rf.fit(X_scaled, y)
y_pred = rf.predict(X_scaled)  # train set just for the report structure
print(f"\n  Full-dataset classification report (for class breakdown, not generalization):")

# Better: do a single held-out fold for the report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
rf2.fit(X_train, y_train)
y_pred_test = rf2.predict(X_test)
print(classification_report(y_test, y_pred_test, digits=3))

# ══════════════════════════════════════════════════════════════
# PART 3: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 3: FEATURE IMPORTANCE (permutation-based)")
print("=" * 70)

perm_imp = permutation_importance(rf2, X_test, y_test, n_repeats=10, random_state=42)
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std
}).sort_values('importance', ascending=False)

for _, row in imp_df.iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"  {row['feature']:25s}: {row['importance']:.4f} ± {row['std']:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════
# PART 4: CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Cluster-Level Feature Separability Analysis\n(Airbus Hackathon - Lidar Obstacle Detection)',
             fontsize=16, fontweight='bold', y=0.98)

colors = {'Antenna': '#2617B4', 'Cable': '#B18430', 'Electric Pole': '#815161', 'Wind Turbine': '#428409'}
class_order = ['Antenna', 'Cable', 'Electric Pole', 'Wind Turbine']

# 1. Height distribution
ax = axes[0, 0]
for cls in class_order:
    subset = df[df['class_label'] == cls]['height']
    ax.hist(subset, bins=40, alpha=0.5, label=cls, color=colors[cls], density=True)
ax.set_xlabel('Bounding Box Height (m)')
ax.set_ylabel('Density')
ax.set_title('Height Distribution')
ax.legend(fontsize=8)

# 2. Verticality distribution
ax = axes[0, 1]
for cls in class_order:
    subset = df[df['class_label'] == cls]['verticality']
    ax.hist(subset, bins=40, alpha=0.5, label=cls, color=colors[cls], density=True, range=(0, 20))
ax.set_xlabel('Verticality (height / max_horizontal)')
ax.set_ylabel('Density')
ax.set_title('Verticality Distribution')
ax.legend(fontsize=8)

# 3. Elongation distribution
ax = axes[0, 2]
for cls in class_order:
    subset = df[df['class_label'] == cls]['elongation']
    ax.hist(subset, bins=40, alpha=0.5, label=cls, color=colors[cls], density=True, range=(0, 30))
ax.set_xlabel('Elongation (max_horizontal / height)')
ax.set_ylabel('Density')
ax.set_title('Elongation Distribution')
ax.legend(fontsize=8)

# 4. Height vs Max Horizontal scatter
ax = axes[1, 0]
for cls in class_order:
    subset = df[df['class_label'] == cls]
    ax.scatter(subset['max_horizontal'], subset['height'], alpha=0.3, label=cls, color=colors[cls], s=15)
ax.set_xlabel('Max Horizontal Extent (m)')
ax.set_ylabel('Height (m)')
ax.set_title('Height vs. Max Horizontal Extent')
ax.legend(fontsize=8)

# 5. Width vs Length scatter
ax = axes[1, 1]
for cls in class_order:
    subset = df[df['class_label'] == cls]
    ax.scatter(subset['width'], subset['length'], alpha=0.3, label=cls, color=colors[cls], s=15)
ax.set_xlabel('Width (m)')
ax.set_ylabel('Length (m)')
ax.set_title('Width vs. Length (BEV footprint)')
ax.legend(fontsize=8)

# 6. Horizontal aspect ratio
ax = axes[1, 2]
for cls in class_order:
    subset = df[df['class_label'] == cls]['horizontal_aspect']
    ax.hist(subset, bins=40, alpha=0.5, label=cls, color=colors[cls], density=True, range=(0, 50))
ax.set_xlabel('Horizontal Aspect Ratio (max/min horizontal)')
ax.set_ylabel('Density')
ax.set_title('Horizontal Aspect Ratio')
ax.legend(fontsize=8)

# 7. Feature importance bar chart
ax = axes[2, 0]
imp_sorted = imp_df.head(10)
bars = ax.barh(range(len(imp_sorted)), imp_sorted['importance'],
               xerr=imp_sorted['std'], color='steelblue', alpha=0.8)
ax.set_yticks(range(len(imp_sorted)))
ax.set_yticklabels(imp_sorted['feature'], fontsize=9)
ax.set_xlabel('Permutation Importance')
ax.set_title('Top 10 Feature Importances')
ax.invert_yaxis()

# 8. Confusion matrix
ax = axes[2, 1]
cm = confusion_matrix(y_test, y_pred_test, labels=class_order)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
            xticklabels=class_order, yticklabels=class_order)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (% per class)\n25% held-out test set')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

# 9. Box plot of key discriminative features
ax = axes[2, 2]
# Show verticality per class as boxplot
box_data = [df[df['class_label'] == cls]['verticality'].clip(0, 25) for cls in class_order]
bp = ax.boxplot(box_data, labels=class_order, patch_artist=True)
for patch, cls in zip(bp['boxes'], class_order):
    patch.set_facecolor(colors[cls])
    patch.set_alpha(0.6)
ax.set_ylabel('Verticality (clipped at 25)')
ax.set_title('Verticality by Class')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('./plots/separability_analysis.png', dpi=150, bbox_inches='tight')
print("\n[Plot saved to separability_analysis.png]")

# ══════════════════════════════════════════════════════════════
# PART 5: CAN A SIMPLE DECISION TREE DO IT? (interpretability)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: HOW SIMPLE CAN WE GO?")
print("=" * 70)

from sklearn.tree import DecisionTreeClassifier
for depth in [2, 3, 5, 8]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight='balanced')
    dt_scores = cross_val_score(dt, X_scaled, y, cv=cv, scoring='f1_weighted')
    print(f"  Decision Tree (depth={depth}): F1={dt_scores.mean():.3f} ± {dt_scores.std():.3f}")

# Logistic regression baseline
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='f1_weighted')
print(f"  Logistic Regression:        F1={lr_scores.mean():.3f} ± {lr_scores.std():.3f}")

# Gradient boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring='f1_weighted')
print(f"  Gradient Boosting:          F1={gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
rf_f1 = f1_scores.mean()
if rf_f1 > 0.85:
    print(f"  ✅ STRONG: RF F1={rf_f1:.3f} — Cluster-level features ARE highly discriminative.")
    print(f"     A simple classifier on bbox geometry can separate these classes well.")
elif rf_f1 > 0.70:
    print(f"  ⚠️  MODERATE: RF F1={rf_f1:.3f} — Some classes separate, others overlap.")
    print(f"     You'll need additional features (point distribution, reflectivity stats).")
else:
    print(f"  ❌ WEAK: RF F1={rf_f1:.3f} — Bbox geometry alone is NOT enough.")
    print(f"     You need richer per-point features or a proper 3D architecture.")