import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
import time

# Same dataset as all C++ experiments
X_train_raw, y_train = make_moons(n_samples=200, noise=0.1,
                                   random_state=42)
X_test_raw,  y_test  = make_moons(n_samples=100, noise=0.1,
                                   random_state=123)

scaler = MinMaxScaler(feature_range=(-1,1))
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

results = []

# 1. Classical RBF kernel SVM (exact)
t0 = time.time()
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
acc_rbf = svm_rbf.score(X_test, y_test)
t_rbf = time.time()-t0
results.append(("Classical RBF SVM (exact)", acc_rbf, t_rbf))

# 2. Random Fourier Features — vary D (number of features)
for D in [4, 8, 16, 32, 64]:
    t0 = time.time()
    rff = Pipeline([
        ('rff', RBFSampler(gamma=1.0, n_components=D,
                           random_state=42)),
        ('svm', SVC(kernel='linear', C=1.0))
    ])
    rff.fit(X_train, y_train)
    acc = rff.score(X_test, y_test)
    t = time.time()-t0
    results.append((f"RFF SVM (D={D})", acc, t))

# 3. Classical polynomial kernel
t0 = time.time()
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train, y_train)
acc_poly = svm_poly.score(X_test, y_test)
t_poly = time.time()-t0
results.append(("Classical Poly-3 SVM", acc_poly, t_poly))

# Print results
print("=== Dequantization Experiment ===")
print(f"{'Method':<30} {'Accuracy':>10} {'Time':>10}")
print("-"*52)
for name, acc, t in results:
    print(f"{name:<30} {acc:>10.3f} {t:>9.4f}s")

print("\n--- Reference (from libqc C++ experiments) ---")
print(f"{'QFT Kernel SVM':<30} {'0.990':>10} {'0.093s':>10}")
print(f"{'RY Kernel SVM':<30} {'0.920':>10} {'0.090s':>10}")
print(f"{'MLP baseline':<30} {'0.800':>10} {'0.002s':>10}")

print("\nKey question:")
print("Does any classical method match QFT kernel (0.99)?")
print("If yes → dequantization. If no → potential quantum advantage.")

# Save CSV
with open("dequantization_results.csv","w") as f:
    f.write("method,accuracy,time\n")
    for name,acc,t in results:
        f.write(f"{name},{acc:.4f},{t:.4f}\n")
    f.write("QFT_kernel_SVM,0.990,0.093\n")
    f.write("RY_kernel_SVM,0.920,0.090\n")
