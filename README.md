# libqc — Quantum ML Simulator in C++20

Custom quantum computing simulator and comparison of 8 QML architectures.
No external dependencies. Built from scratch.

## Algorithms implemented
- Standard QNN (parameter shift gradient)
- Data Reuploading QNN (Pérez-Salinas 2020)
- Quantum Kernel SVM (fidelity kernel)
- QFT Kernel SVM (Fourier feature map)
- Quantum Reservoir Computing + Ridge Regression
- VQE — H₂ ground state (chemical accuracy)
- Potential Energy Curve (PEC)
- Barren Plateaus diagnostic
- Noise Model (depolarizing channel)

## Results on make_moons
| Method | Accuracy |
|--------|----------|
| QFT Kernel SVM | 99% |
| Reuploading 4L | 95% |
| QK-SVM | 92% |
| QRC (n=10) | 89% |
| MLP baseline | 80% |
| Standard QNN | 62% |

## Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/experiment8

## Run all experiments
./build/experiment   # QNN vs MLP
./build/experiment2  # Quantum Kernel SVM
./build/experiment3  # VQE
./build/experiment4  # Barren Plateaus
./build/experiment5  # PEC
./build/experiment6  # Data Reuploading
./build/experiment7  # QRC
./build/experiment8  # QFT Kernel
./build/experiment9  # Noise Model

## Python experiments
pip install scikit-learn pennylane qiskit
python experiments/dequantization.py
python experiments/dequantization_qrc.py
