import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
import time

np.random.seed(42)

X_train_raw, y_train = make_moons(n_samples=200, noise=0.1,
                                   random_state=42)
X_test_raw,  y_test  = make_moons(n_samples=100, noise=0.1,
                                   random_state=123)

scaler = MinMaxScaler(feature_range=(-1,1))
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

results = []

# 1. Classical ESN — vary reservoir size n
def esn_features(X, n=10, spectral_radius=0.9, seed=42):
    rng = np.random.RandomState(seed)
    W = rng.randn(n, n)
    # scale to desired spectral radius
    W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W)))
    W_in = rng.randn(n, X.shape[1])
    features = []
    for x in X:
        h = np.zeros(n)
        h = np.tanh(W @ h + W_in @ x)
        features.append(h)
    return np.array(features)

for n in [4, 6, 8, 10, 20]:
    t0 = time.time()
    F_train = esn_features(X_train, n=n)
    F_test  = esn_features(X_test,  n=n)
    clf = RidgeClassifier(alpha=1e-3)
    clf.fit(F_train, y_train)
    acc = clf.score(F_test, y_test)
    t = time.time()-t0
    results.append((f"Classical ESN (n={n})", acc, t))

# 2. Random Kitchen Sinks — random cosine features
def rks_features(X, D=10, sigma=1.0, seed=42):
    rng = np.random.RandomState(seed)
    omega = rng.randn(D, X.shape[1]) / sigma
    b     = rng.uniform(0, 2*np.pi, D)
    return np.sqrt(2/D) * np.cos(X @ omega.T + b)

for D in [4, 6, 8, 10, 20]:
    t0 = time.time()
    F_train = rks_features(X_train, D=D)
    F_test  = rks_features(X_test,  D=D)
    clf = RidgeClassifier(alpha=1e-3)
    clf.fit(F_train, y_train)
    acc = clf.score(F_test, y_test)
    t = time.time()-t0
    results.append((f"Classical RKS (D={D})", acc, t))

# 3. MLP baseline
from sklearn.neural_network import MLPClassifier
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=100,
                    random_state=42)
mlp.fit(X_train, y_train)
acc_mlp = mlp.score(X_test, y_test)
results.append(("Classical MLP", acc_mlp, time.time()-t0))

# Print
print("=== Dequantization: QRC vs Classical ===\n")
print(f"{'Method':<30} {'Accuracy':>10} {'Time':>10}")
print("-"*52)
for name, acc, t in results:
    print(f"{name:<30} {acc:>10.3f} {t:>9.4f}s")

print("\n--- Reference (libqc QRC) ---")
print(f"{'QRC n=4':<30} {'0.860':>10}")
print(f"{'QRC n=6':<30} {'0.790':>10}")
print(f"{'QRC n=8':<30} {'0.770':>10}")
print(f"{'QRC n=10':<30} {'0.890':>10}")

print("\nVerdict: does classical reservoir match quantum reservoir?")

with open("dequantization_qrc_results.csv","w") as f:
    f.write("method,accuracy,time\n")
    for name,acc,t in results:
        f.write(f"{name},{acc:.4f},{t:.4f}\n")
