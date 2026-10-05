#include "Matrix.h"
#include "QuantumGate.h"
#include "QuantumState.h"
#include "QuantumCircuit.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <numbers>

static void check(bool cond, const char* label) {
    std::cout << (cond ? "[OK] " : "[FAIL] ") << label << "\n";
}

// ── Matrix ────────────────────────────────────────────────────────────────────

static void test_matrix() {
    std::cout << "\n=== Matrix ===\n";

    Matrix id(3, 3);
    id(0,0) = id(1,1) = id(2,2) = Complex(1.0, 0.0);
    check(id.IsIdentity(), "3x3 identity recognized");

    Matrix notId(2, 2);
    notId(0,0) = Complex(2.0, 0.0);
    check(!notId.IsIdentity(), "non-identity rejected");

    Matrix a(2, 2), b(2, 2);
    a(0,0) = Complex(1,0); a(0,1) = Complex(2,0);
    a(1,0) = Complex(3,0); a(1,1) = Complex(4,0);
    b(0,0) = Complex(5,0); b(0,1) = Complex(6,0);
    b(1,0) = Complex(7,0); b(1,1) = Complex(8,0);
    Matrix c = a.Multiply(b);
    check(c(0,0) == Complex(19,0) && c(1,1) == Complex(50,0), "2x2 multiply");

    std::cout << "Matrix a:\n" << a;
}

// ── QuantumGate ───────────────────────────────────────────────────────────────

static void test_gates() {
    std::cout << "\n=== QuantumGate ===\n";

    check(QuantumGate::X().matrix().IsUnitary(),    "X is unitary");
    check(QuantumGate::H().matrix().IsUnitary(),    "H is unitary");
    check(QuantumGate::RZ(1.23).matrix().IsUnitary(), "RZ(1.23) is unitary");
    check(QuantumGate::CNOT().matrix().IsUnitary(), "CNOT matrix is unitary");

    // X|0> = |1>
    QuantumState s(1);
    s.apply_single_qubit(QuantumGate::X(), 0);
    check(std::abs(s[0]) < 1e-9 && std::abs(s[1] - Complex(1,0)) < 1e-9, "X flips |0> to |1>");

    // H|0> = (|0>+|1>)/sqrt(2)
    QuantumState h(1);
    h.apply_single_qubit(QuantumGate::H(), 0);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    check(std::abs(h[0] - Complex(inv_sqrt2,0)) < 1e-9 &&
          std::abs(h[1] - Complex(inv_sqrt2,0)) < 1e-9, "H creates superposition");
}

// ── Bell state ────────────────────────────────────────────────────────────────

static void test_bell() {
    std::cout << "\n=== Bell state (H⊗I + CNOT) ===\n";

    QuantumState bell(2);
    bell.apply_single_qubit(QuantumGate::H(), 0);
    bell.apply_cnot(0, 1);

    // Stan powinien być (|00> + |11>) / sqrt(2)
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    check(std::abs(bell[0] - Complex(inv_sqrt2, 0)) < 1e-9, "Bell: amplitude |00>");
    check(std::abs(bell[1]) < 1e-9,                          "Bell: amplitude |01> = 0");
    check(std::abs(bell[2]) < 1e-9,                          "Bell: amplitude |10> = 0");
    check(std::abs(bell[3] - Complex(inv_sqrt2, 0)) < 1e-9, "Bell: amplitude |11>");
    check(bell.isNormalized(), "Bell state normalized");

    // <Z> na kubicie 0 w stanie Bella = 0
    double ez = bell.expectation_z(0);
    check(std::abs(ez) < 1e-9, "Bell: <Z0> = 0");
}

// ── QuantumCircuit ────────────────────────────────────────────────────────────

static void test_circuit() {
    std::cout << "\n=== QuantumCircuit ===\n";

    // Odtworzenie stanu Bella przez obwód
    QuantumCircuit c(2);
    c.add_gate(QuantumGate::H(), 0);
    c.add_cnot(0, 1);

    QuantumState s(2);
    c.run(s);

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    check(std::abs(s[0] - Complex(inv_sqrt2, 0)) < 1e-9 &&
          std::abs(s[3] - Complex(inv_sqrt2, 0)) < 1e-9, "Circuit reproduces Bell state");

    // Parametryczny RZ
    QuantumCircuit rz_circ(1);
    size_t p0 = rz_circ.add_rz(0.0, 0);  // RZ(0) = I na fazie globalnej
    check(rz_circ.num_parameters() == 1, "Circuit has 1 parameter");

    rz_circ.set_parameter(p0, std::numbers::pi);  // RZ(pi)
    QuantumState rs(1);
    rz_circ.run(rs);
    // RZ(pi)|0> = e^{-i*pi/2}|0> — norma bez zmian
    check(rs.isNormalized(), "Parametric RZ preserves normalization");
}

// ── Parameter shift ───────────────────────────────────────────────────────────

static void test_parameter_shift() {
    std::cout << "\n=== Parameter shift ===\n";

    // Obwód: H → RZ(theta) → H na |0>.
    // <Z0> = cos(theta),  d/dtheta = -sin(theta).
    auto run_circuit = [](double theta) -> double {
        QuantumCircuit c(1);
        c.add_gate(QuantumGate::H(), 0);
        c.add_rz(theta, 0);
        c.add_gate(QuantumGate::H(), 0);
        QuantumState s(1);
        c.run(s);
        return s.expectation_z(0);
    };

    const double theta = 0.7;
    const double shift = std::numbers::pi / 2.0;

    double grad_numeric = (run_circuit(theta + shift) - run_circuit(theta - shift)) / 2.0;
    double grad_exact   = -std::sin(theta);   // analityczna pochodna cos(theta)

    check(std::abs(grad_numeric - grad_exact) < 1e-9, "Parameter shift gradient correct");
}

// ── measure() ─────────────────────────────────────────────────────────────────

static void test_measure() {
    std::cout << "\n=== measure() ===\n";

    // Stan |1> — pomiar zawsze daje 1
    QuantumState s(1);
    s.apply_single_qubit(QuantumGate::X(), 0);
    size_t result = s.measure();
    check(result == 1, "Measuring |1> always gives 1");

    // Po kolapsie stan znowu znormalizowany
    check(s.isNormalized(), "State normalized after measurement");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_matrix();
    test_gates();
    test_bell();
    test_circuit();
    test_parameter_shift();
    test_measure();
    std::cout << "\nDone.\n";
    return 0;
}
