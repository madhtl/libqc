from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUTPUT = "/Users/madzia/CLionProjects/libqc/algorytmy_libqc.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
)

W = A4[0] - 4*cm

# ── style ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "MyTitle",
    fontSize=22, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a237e"),
    spaceAfter=6, alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "MySubtitle",
    fontSize=11, fontName="Helvetica",
    textColor=colors.HexColor("#455a64"),
    spaceAfter=16, alignment=TA_CENTER,
)
algo_title_style = ParagraphStyle(
    "AlgoTitle",
    fontSize=14, fontName="Helvetica-Bold",
    textColor=colors.white,
    spaceAfter=0, spaceBefore=0,
    leftIndent=8,
)
section_head_style = ParagraphStyle(
    "SectionHead",
    fontSize=10, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a237e"),
    spaceBefore=8, spaceAfter=3,
)
body_style = ParagraphStyle(
    "Body",
    fontSize=9, fontName="Helvetica",
    leading=13, spaceAfter=2,
    alignment=TA_JUSTIFY,
)
code_style = ParagraphStyle(
    "Code",
    fontSize=8, fontName="Courier",
    textColor=colors.HexColor("#263238"),
    leading=11, spaceAfter=1,
    backColor=colors.HexColor("#eceff1"),
    leftIndent=6, rightIndent=6,
    borderPad=4,
)
step_style = ParagraphStyle(
    "Step",
    fontSize=9, fontName="Helvetica",
    leading=13, spaceAfter=1,
    leftIndent=12,
)
fn_style = ParagraphStyle(
    "Fn",
    fontSize=8.5, fontName="Courier-Bold",
    textColor=colors.HexColor("#b71c1c"),
    leading=12, spaceAfter=1,
    leftIndent=6,
)

PALETTE = [
    colors.HexColor("#37474f"),  # 1 MLP
    colors.HexColor("#1565c0"),  # 2 QNN
    colors.HexColor("#283593"),  # 3 VQE
    colors.HexColor("#4527a0"),  # 4 PEC
    colors.HexColor("#6a1b9a"),  # 5 Reupload
    colors.HexColor("#00695c"),  # 6 QKernel
    colors.HexColor("#00838f"),  # 7 QFTKernel
    colors.HexColor("#e65100"),  # 8 Reservoir
    colors.HexColor("#c62828"),  # 9 Barren
]

def algo_block(number, title, subtitle, color, functions, steps, note=None):
    elems = []

    # ── colored header bar ────────────────────────────────────────────────────
    header_data = [[
        Paragraph(f"<b>{number}.</b>  {title}", algo_title_style),
        Paragraph(f"<i>{subtitle}</i>",
                  ParagraphStyle("sub2", fontSize=9, fontName="Helvetica-Oblique",
                                 textColor=colors.HexColor("#cfd8dc"), alignment=4)),
    ]]
    header_table = Table(header_data, colWidths=[W*0.62, W*0.38])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",  (0,0), (0,-1), 10),
        ("RIGHTPADDING", (-1,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4,4,0,0]),
    ]))
    elems.append(header_table)

    # ── inner content box ─────────────────────────────────────────────────────
    inner = []

    # functions
    inner.append(Paragraph("Zaangażowane funkcje / klasy:", section_head_style))
    for fn in functions:
        inner.append(Paragraph(fn, fn_style))
    inner.append(Spacer(1, 6))

    # steps
    inner.append(Paragraph("Dokładne postępowanie:", section_head_style))
    for i, step in enumerate(steps, 1):
        inner.append(Paragraph(f"<b>{i}.</b>  {step}", step_style))
    if note:
        inner.append(Spacer(1, 4))
        inner.append(Paragraph(f"<b>Uwaga:</b> {note}", body_style))

    inner_table = Table([[inner]], colWidths=[W])
    inner_table.setStyle(TableStyle([
        ("BOX",          (0,0), (-1,-1), 1,   color),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#fafafa")),
    ]))
    elems.append(inner_table)
    elems.append(Spacer(1, 14))
    return elems


story = []

# ── COVER ─────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("libqc — przegląd algorytmów", title_style))
story.append(Paragraph(
    "Od najprostszego do najbardziej rozbudowanego · z dokładnym wykazem funkcji i kroków postępowania",
    subtitle_style))
story.append(HRFlowable(width=W, thickness=1.5, color=colors.HexColor("#1a237e")))
story.append(Spacer(1, 0.6*cm))

intro = (
    "Dokument prezentuje dziewięć modeli zaimplementowanych w bibliotece <b>libqc</b> (C++17) "
    "w kolejności rosnącej złożoności — od klasycznej sieci MLP stanowiącej punkt odniesienia, "
    "przez wariantowe klasyfikatory kwantowe (QNN, Data Reuploading), metody jądrowe "
    "(QuantumKernel + SVM, QFTKernel + SVM), obliczenia chemii kwantowej (VQE, PEC), "
    "Quantum Reservoir Computing, aż po diagnostyczny eksperyment Barren Plateau. "
    "Dla każdego algorytmu podane są: zaangażowane klasy i funkcje (z sygnaturami) "
    "oraz krok-po-kroku opis postępowania."
)
story.append(Paragraph(intro, body_style))
story.append(Spacer(1, 0.5*cm))

# ── ALGO 1: MLP ───────────────────────────────────────────────────────────────
story += algo_block(
    1, "MLP — Klasyczna sieć neuronowa", "src/MLP.cpp · include/MLP.h", PALETTE[0],
    functions=[
        "MLP::MLP()",
        "double MLP::forward(double x1, double x2)",
        "FwdCache MLP::forward_cached(double x1, double x2) const",
        "int MLP::predict(double x1, double x2)",
        "void MLP::train(vector<Sample>, int epochs, double lr, ...)",
        "double MLP::accuracy(vector<Sample>)",
        "static double MLP::sigmoid(double x)",
    ],
    steps=[
        "<b>Inicjalizacja</b> — <font face='Courier'>MLP()</font>: wagi W1 (8 liczb = macierz 4×2), b1 (4), W2 (4), b2 losowane z U(−0.5, 0.5) przez mt19937(seed=99).",
        "<b>Forward pass</b> — <font face='Courier'>forward_cached(x1,x2)</font>: dla każdego neuronu j∈{0..3} liczy net_j = Σ_i W1[j·2+i]·x_i + b1[j], następnie h1[j] = sigmoid(net_j). Wyjście: net2 = Σ_j W2[j]·h1[j] + b2, y = sigmoid(net2).",
        "<b>Backprop (jedna próbka)</b> — strata MSE: L=(y−label)². Delta warstwy wyjściowej: δ₂ = 2(y−label)·y·(1−y). Delta ukrytej: δ₁[j] = W2[j]·δ₂·h1[j]·(1−h1[j]).",
        "<b>Akumulacja gradientów przez epokę</b> — dW2[j] += δ₂·h1[j]; db2 += δ₂; dW1[j·2+i] += δ₁[j]·x_i; db1[j] += δ₁[j].",
        "<b>Aktualizacja wag</b> — po wszystkich próbkach: param −= lr · grad / n.",
        "<b>Predykcja</b> — <font face='Courier'>predict()</font>: forward > 0.5 → klasa 1, wpp → klasa 0.",
    ],
    note="Architektura stała: 2→4→1, aktywacja sigmoid wszędzie, loss MSE."
)

# ── ALGO 2: QNN ───────────────────────────────────────────────────────────────
story += algo_block(
    2, "QNN — Podstawowy klasyfikator kwantowy", "src/QNN.cpp · include/QNN.h", PALETTE[1],
    functions=[
        "QNN::QNN(int n_layers)",
        "double QNN::forward(double x1, double x2)",
        "int QNN::predict(double x1, double x2)",
        "double QNN::param_shift_grad(double x1, double x2, int trainable_idx)",
        "void QNN::train(vector<Sample>, int epochs, double lr, ...)",
        "double QNN::accuracy(vector<Sample>)",
        "— używa: QuantumCircuit, QuantumState, QuantumGate::RY/CNOT",
    ],
    steps=[
        "<b>Budowa obwodu</b> — <font face='Courier'>QNN(n_layers)</font>: 2-kubitowy QuantumCircuit. Najpierw 2 RY (param[0]=x1π, param[1]=x2π — enkodowanie danych). Następnie n_layers razy: RY(θ,q0), RY(θ,q1), CNOT(0→1). Łącznie 2+2·L parametrów; param[0..1] to dane, param[2..1+2L] to wagi.",
        "<b>Forward pass</b> — <font face='Courier'>forward(x1,x2)</font>: ustaw param[0]=x1·π, param[1]=x2·π; stwórz stan |00⟩; wywołaj circuit.run(s) → aplikuje kolejne bramki; zwróć s.expectation_z(0) = Σᵢ eigenvalue(i,q=0)·|aᵢ|².",
        "<b>Gradient przez parameter shift</b> — <font face='Courier'>param_shift_grad(x1,x2,i)</font>: przesuń θᵢ o +π/2, wywołaj forward, zapamiętaj f⁺; przesuń o −π/2, wywołaj forward, zapamiętaj f⁻; przywróć θᵢ; zwróć (f⁺−f⁻)/2.",
        "<b>Pętla treningowa</b> — dla każdej epoki: akumuluj gradient MSE: grad[i] += 2·(y−label_±1)·param_shift_grad(x1,x2,i). Po epoce: θᵢ −= lr·grad[i]/n.",
        "<b>Predykcja</b> — expectation_z(0) > 0 → klasa 1, wpp → klasa 0.",
    ],
    note="Enkodowanie danych następuje tylko raz na początku obwodu — dane nie są 're-uploadowane' w głębszych warstwach (to ograniczenie poprawia QNN_Reupload)."
)

# ── ALGO 3: VQE ───────────────────────────────────────────────────────────────
story += algo_block(
    3, "VQE — Variational Quantum Eigensolver", "src/VQE.cpp · include/VQE.h", PALETTE[2],
    functions=[
        "PauliHamiltonian::add_term(double coeff, string ops)",
        "double PauliHamiltonian::pauli_expectation(QuantumState state, const string& ops) const",
        "double PauliHamiltonian::expectation(QuantumState state) const",
        "static PauliHamiltonian PauliHamiltonian::H2()",
        "QuantumState VQE::ansatz(const vector<double>& theta) const",
        "double VQE::energy(const vector<double>& theta)",
        "VQEResult VQE::optimize(int n_steps=200, double lr=0.1)",
    ],
    steps=[
        "<b>Hamiltonian H₂</b> — <font face='Courier'>H2()</font> buduje 6-wyrazowy operator Pauliego (Jordan-Wigner, STO-3G, R=0.735 Å): −0.4804·II + 0.3435·ZI − 0.4347·IZ + 0.5716·ZZ + 0.0910·YY + 0.0910·XX.",
        "<b>Ansatz HEA</b> — <font face='Courier'>ansatz(θ)</font>: |00⟩ → RY(θ₀,q0) → RY(θ₁,q1) → CNOT(0→1) → RY(θ₂,q0) → RY(θ₃,q1). Cztery parametry, obwód 2-kubitowy.",
        "<b>Wartość oczekiwana</b> — <font face='Courier'>pauli_expectation(state, ops)</font>: kopia stanu; dla każdego qubitu q aplikuj X/Y/Z lub I; oblicz Re(Σᵢ conj(ψ[i])·(P|ψ⟩)[i]). <font face='Courier'>expectation()</font> sumuje coeff_k · pauli_expectation dla każdego termu.",
        "<b>Optymalizacja</b> — <font face='Courier'>optimize(200, 0.1)</font>: inicjalizuj θ = [0.1, 0.1, 0.1, 0.1]; w każdym kroku dla każdego θᵢ oblicz gradient parameter-shift: grad[i]=(E(θᵢ+π/2)−E(θᵢ−π/2))/2; zaktualizuj θᵢ −= lr·grad[i]; zapisz E do historii.",
        "<b>Wynik</b> — zwróć VQEResult{energy, theta, history}; energy to dolna granica energii stanu podstawowego.",
    ],
    note="Cel VQE: minimalizacja ⟨ψ(θ)|H|ψ(θ)⟩. Energia dokładna H₂ w tym modelu ≈ −1.137 Ha."
)

# ── ALGO 4: PEC ───────────────────────────────────────────────────────────────
story += algo_block(
    4, "PEC — Krzywa potencjału (rozszerzenie VQE)", "src/PEC.cpp · include/PEC.h", PALETTE[3],
    functions=[
        "PECCalculator::PECCalculator()   // tabela koeficjentów Pauliego",
        "PauliHamiltonian PECCalculator::hamiltonian_at(double R) const",
        "double PECCalculator::energy_at(double R)   // jeden VQE",
        "vector<PECPoint> PECCalculator::compute_all()   // pętla po R",
        "PECMinimum PECCalculator::find_minimum(vector<PECPoint>)",
        "— wewnętrznie: VQE::optimize(), PauliHamiltonian::expectation()",
    ],
    steps=[
        "<b>Tabela koeficjentów</b> — konstruktor wypełnia mapę <font face='Courier'>coeff[R] = {g0,..,g5}</font> dla 12 odległości R ∈ {0.50, 0.60, …, 2.50} Å. Wartości {g0..g5} odpowiadają wkładom II, ZI, IZ, ZZ, XX, YY (obliczone z PySCF/RHF/STO-3G).",
        "<b>Hamiltonian dla R</b> — <font face='Courier'>hamiltonian_at(R)</font>: lookup w mapie (brak R → wyjątek); buduje PauliHamiltonian z 6 termami używając odczytanych koeficjentów.",
        "<b>Energia dla jednego R</b> — <font face='Courier'>energy_at(R)</font>: tworzy VQE(hamiltonian_at(R), 2) i wywołuje optimize(1000, 0.1). Zwraca energię optymalną.",
        "<b>Cała krzywa</b> — <font face='Courier'>compute_all()</font>: iteruje po wszystkich kluczach mapy i wywołuje energy_at(R) dla każdego, zbiera PECPoint{R, energy, 0}.",
        "<b>Minimum</b> — <font face='Courier'>find_minimum()</font>: szuka indeksu min energii; R_e = R przy minimum; De = E(R=0.50) − E_min (przybliżona energia dysocjacji).",
    ],
    note="PEC to 12 niezależnych uruchomień VQE (1000 kroków każde). Najdroższy obliczeniowo moduł."
)

# ── ALGO 5: QNN_Reupload ──────────────────────────────────────────────────────
story += algo_block(
    5, "QNN Reupload — Data Reuploading (Pérez-Salinas 2020)", "src/QNN_Reupload.cpp · include/QNN_Reupload.h", PALETTE[4],
    functions=[
        "QNN_Reupload::QNN_Reupload(int n_layers)",
        "double QNN_Reupload::forward(double x1, double x2)",
        "int QNN_Reupload::predict(double x1, double x2)",
        "double QNN_Reupload::param_shift_grad(double x1, double x2, int trainable_idx)",
        "void QNN_Reupload::train(vector<Sample>, int epochs, double lr, ...)",
        "double QNN_Reupload::accuracy(vector<Sample>)",
    ],
    steps=[
        "<b>Budowa obwodu</b> — <font face='Courier'>QNN_Reupload(n_layers)</font>: dla każdej warstwy l dodaje 4 parametry: param[4l+0]=x1π (data q0), param[4l+1]=x2π (data q1), param[4l+2]=θ (train q0), param[4l+3]=θ (train q1); po nich CNOT(0→1). Łącznie 4·L parametrów.",
        "<b>Forward pass</b> — <font face='Courier'>forward(x1,x2)</font>: dla każdej warstwy l ustaw param[4l+0]=x1π i param[4l+1]=x2π (re-upload danych). Stwórz |00⟩, wywołaj circuit.run(s), zwróć s.expectation_z(0).",
        "<b>Kluczowa różnica vs QNN</b> — dane x1, x2 są wstrzykiwane ponownie PRZED każdą warstwą treningową, a nie tylko raz na początku. Daje to silniejszą ekspresywność.",
        "<b>Gradient</b> — <font face='Courier'>param_shift_grad(x1,x2,i)</font>: indeksowanie: idx = 4·(i÷2) + 2 + (i mod 2), tzn. omija param danych i trafia tylko w wagi treningowe. Reszta identyczna jak w QNN.",
        "<b>Pętla treningowa</b> — identyczna z QNN: MSE loss, akumulacja gradientów, aktualizacja θ −= lr·grad/n. Pętla po trainable_idx obejmuje tylko parametry treningowe (pomija data-params).",
    ],
    note="Jeden 'neuron kwantowy' z re-uploadem może aproksymować dowolną funkcję Fouriera — teoretycznie silniejszy od jednorazowego enkodowania."
)

# ── ALGO 6: QuantumKernel + SVM ───────────────────────────────────────────────
story += algo_block(
    6, "QuantumKernel + SVM — Jądro fidelity", "src/QuantumKernel.cpp · src/SVM.cpp", PALETTE[5],
    functions=[
        "QuantumKernel::QuantumKernel(int n_qubits=4)",
        "QuantumState QuantumKernel::encode(double x1, double x2) const",
        "double QuantumKernel::kernel(double x1i,x2i, double x1j,x2j) const",
        "vector<vector<double>> QuantumKernel::kernel_matrix(vector<Sample>) const",
        "vector<vector<double>> QuantumKernel::kernel_matrix(test, train) const",
        "static double QuantumKernel::inner_product(QuantumState a, b)",
        "KernelSVM::KernelSVM(double C=1.0, double lr=1e-2)",
        "void KernelSVM::fit(vector<vector<double>> K, vector<int> labels, int n_iter=1000)",
        "vector<int> KernelSVM::predict(vector<vector<double>> K_test) const",
        "int KernelSVM::num_support_vectors() const",
    ],
    steps=[
        "<b>Enkodowanie</b> — <font face='Courier'>encode(x1,x2)</font>: 4-kubitowy stan startowy |0000⟩. Warstwa 1: H na każdym kubicie. Warstwa 2: RY(x1π,q0), RY(x2π,q1), RY(x1π,q2), RY(x2π,q3). Splątanie: CNOT(0→1), CNOT(2→3). Warstwa 4: ponownie RY (te same kąty). Zwraca QuantumState z 16 amplitudami.",
        "<b>Jądro fidelity</b> — <font face='Courier'>kernel(xᵢ,xⱼ)</font>: wywołuje encode dwukrotnie; <font face='Courier'>inner_product(a,b)</font> = |Σᵢ conj(a[i])·b[i]|² = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|².",
        "<b>Macierz jądra</b> — <font face='Courier'>kernel_matrix(data)</font>: n×n symetryczna macierz, K[i][i]=1, K[i][j]=K[j][i]=kernel(...). Złożoność O(n²) enkodowań.",
        "<b>Trening SVM (problem dualny)</b> — <font face='Courier'>fit(K,labels,1000)</font>: inicjalizuj α=0; w każdej iteracji dla każdego i: decision = Σⱼ αⱼ·yⱼ·K[i][j]; grad = 1 − yᵢ·decision; αᵢ = clip(αᵢ + lr·grad, 0, C). Bias obliczany jako średnia z support vectors.",
        "<b>Predykcja</b> — <font face='Courier'>predict(K_test)</font>: score = bias + Σⱼ αⱼ·yⱼ·K_test[i][j]; klasa = sign(score).",
    ],
    note="Jądro kwantowe jest PSD z definicji (jako moduł iloczynu skalarnego). SVM nie rozwiązuje pełnego SMO — uproszczony gradient ascent wystarczy do porównania."
)

# ── ALGO 7: QFTKernel + SVM ───────────────────────────────────────────────────
story += algo_block(
    7, "QFTKernel + SVM — Jądro z transformatą Fouriera", "src/QFTKernel.cpp · include/QFTKernel.h", PALETTE[6],
    functions=[
        "QFTKernel::QFTKernel(int n_qubits=4)",
        "QuantumState QFTKernel::encode(double x1, double x2) const",
        "double QFTKernel::kernel(double x1i,x2i, double x1j,x2j) const",
        "vector<vector<double>> QFTKernel::kernel_matrix(vector<Sample>) const",
        "vector<vector<double>> QFTKernel::kernel_matrix_test(test, train) const",
        "static double QFTKernel::inner_product(QuantumState a, b)",
        "— SVM identyczny jak wyżej: KernelSVM::fit / predict",
    ],
    steps=[
        "<b>Enkodowanie kątowe</b> — <font face='Courier'>encode(x1,x2)</font> krok 1: RY(x1π,q0), RY(x2π,q1), następnie dla q∈{2,3,...}: RY(x1π,qEven), RY(x2π,qOdd). Brak warstwy H na wejściu — zaczyna od stanu |0…0⟩.",
        "<b>QFT (bez bit reversal)</b> — krok 2: dla i=0..n−1: aplikuj H(i); dla j=i+1..n−1: <font face='Courier'>apply_controlled_phase(2π/2^(j−i+1), i, j)</font>. To standardowy QFT bez końcowego zamiany bitów — operuje na przestrzeni Hilberta strukturyzowanej przez bazy Fouriera.",
        "<b>Jądro fidelity</b> — identyczne z QuantumKernel: K(xᵢ,xⱼ) = |⟨ψ_QFT(xᵢ)|ψ_QFT(xⱼ)⟩|².",
        "<b>Macierze jądra</b> — <font face='Courier'>kernel_matrix(data)</font> = n×n (trening); <font face='Courier'>kernel_matrix_test(test,train)</font> = n_test×n_train (predykcja). Pełna O(n²) pętla.",
        "<b>SVM</b> — identyczny algorytm jak w QuantumKernel: fit z K_train, predict z K_test.",
    ],
    note="Różnica vs QuantumKernel: QFT wprowadza wieloskalową strukturę Fouriera do przestrzeni cech. Efektywnie sprawia, że jądro jest bardziej wrażliwe na periodyczne wzorce w danych."
)

# ── ALGO 8: QuantumReservoir ──────────────────────────────────────────────────
story += algo_block(
    8, "Quantum Reservoir Computing + Ridge", "src/QuantumReservoir.cpp · include/QuantumReservoir.h", PALETTE[7],
    functions=[
        "QuantumReservoir::QuantumReservoir(int n_qubits=6, unsigned seed=42)",
        "ReservoirFeatures QuantumReservoir::encode(double x1, double x2) const",
        "vector<ReservoirFeatures> QuantumReservoir::encode_all(vector<Sample>) const",
        "int QuantumReservoir::numFeatures() const   // = n_qubits",
        "RidgeClassifier::RidgeClassifier(double lambda=1e-3)",
        "void RidgeClassifier::fit(vector<ReservoirFeatures> X, vector<int> labels)",
        "int RidgeClassifier::predict(ReservoirFeatures x) const",
        "double RidgeClassifier::accuracy(vector<ReservoirFeatures>, vector<int>) const",
    ],
    steps=[
        "<b>Inicjalizacja rezerwuaru</b> — <font face='Courier'>QuantumReservoir(6, 42)</font>: losuje macierz wag weights[3][6] z U(−π, π) przez mt19937(42). Wagi są STAŁE — nigdy nie są aktualizowane.",
        "<b>Ekstrakcja cech</b> — <font face='Courier'>encode(x1,x2)</font>: buduje nowy QuantumCircuit(6) za każdym razem. Krok 1: RY(x1π,q0), RY(x2π,q1), RY(0,q2..5) — wstrzyknięcie danych. Krok 2: 3 warstwy: RY(weights[l][q]) na każdym kubicie, CNOT(q→q+1) łańcuch. Krok 3: oblicz ⟨Z_q⟩ dla q=0..5 → 6-wymiarowy wektor cech.",
        "<b>Enkodowanie całego zbioru</b> — <font face='Courier'>encode_all(data)</font>: wywołuje encode dla każdej próbki, zwraca wektor ReservoirFeatures.",
        "<b>Ridge regression (closed-form)</b> — <font face='Courier'>fit(X, labels)</font>: buduje X_aug (n×7, z kolumną biasów = 1). A = X_aug^T·X_aug + λ·I (7×7). b = X_aug^T·y. Rozwiązuje A·w = b przez eliminację Gaussa-Jordana z pivotowaniem częściowym. Wynik: wagi w[6] + bias.",
        "<b>Predykcja</b> — <font face='Courier'>predict(x)</font>: val = bias + Σᵢ w[i]·x.features[i]; > 0.5 → klasa 1.",
    ],
    note="Reservoir nie jest trenowany — jedyne trenowane parametry to wagi ridge. Całe uczenie to jedno rozwiązanie układu równań liniowych, bez iteracji."
)

# ── ALGO 9: Barren Plateau ────────────────────────────────────────────────────
story += algo_block(
    9, "Barren Plateau — Eksperyment diagnostyczny", "src/BarrenPlateau.cpp · include/BarrenPlateau.h", PALETTE[8],
    functions=[
        "double BarrenPlateauExperiment::compute_gradient(int n_qubits, vector<double> theta)",
        "BPResult BarrenPlateauExperiment::run(int n_qubits, int n_samples=200)",
        "vector<BPResult> BarrenPlateauExperiment::run_all(int max_qubits=8, int n_samples=200)",
        "— struct BPResult { n_qubits, mean_gradient, variance_gradient, std_gradient }",
        "— używa bezpośrednio: QuantumState, QuantumGate::RY(), apply_cnot()",
    ],
    steps=[
        "<b>Ansatz diagnostyczny</b> — <font face='Courier'>compute_gradient(n, θ)</font>: L=n warstw, każda: RY(θ[l·n+q], q) dla q=0..n−1; CNOT(q→q+1) łańcuch. Obserwabla: expectation_z(q=0). Łącznie L·n = n² parametrów.",
        "<b>Gradient przez parameter shift</b> — tylko po θ[0] (RY na q=0 w warstwie 0): λ run_circuit(θ₀+π/2) − run_circuit(θ₀−π/2)) / 2. Wszystkie inne θ są STAŁE dla danego losowego zestawu.",
        "<b>Jedno uruchomienie</b> — <font face='Courier'>run(n_qubits, 200)</font>: dla s=0..199: wylosuj θ (mt19937(s·1000+n), U(−π,π)); oblicz gradient. Zbierz 200 wartości.",
        "<b>Statystyki</b> — E[g] = mean, E[g²] = mean_sq, Var[g] = E[g²] − E[g]². Teoria BP: Var[g] ∝ 2^(−n) — wykładnicze zanikanie z liczbą kubitów.",
        "<b>Pełny eksperyment</b> — <font face='Courier'>run_all(8)</font>: iteruje n=2..8; zwraca wektor BPResult. Wynik pokazuje, jak wariancja gradientu spada eksponencjalnie wraz z rozmiarem układu.",
    ],
    note="Barren Plateau to NIE klasyfikator — to narzędzie do badania problemu zaniku gradientów w głębokich obwodach kwantowych (ważne uzasadnienie dla płytkich ansatz-ów jak QNN/Reupload)."
)

# ── PODSUMOWANIE ──────────────────────────────────────────────────────────────
story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor("#37474f")))
story.append(Spacer(1, 8))
story.append(Paragraph("Zestawienie porównawcze", ParagraphStyle(
    "CompHead", fontSize=13, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a237e"), spaceAfter=8
)))

table_data = [
    ["#", "Algorytm", "Kubity", "Parametry", "Trening", "Cel"],
    ["1", "MLP",             "0",  "13",         "backprop",           "klasyfikacja"],
    ["2", "QNN",             "2",  "2+2L",        "param-shift GD",     "klasyfikacja"],
    ["3", "VQE",             "2",  "4",           "param-shift GD",     "chemia kw."],
    ["4", "PEC",             "2",  "4×12",        "12 × VQE",           "krzywa potencjału"],
    ["5", "QNN Reupload",    "2",  "4L (2 train)","param-shift GD",     "klasyfikacja"],
    ["6", "QuantumKernel+SVM","4", "brak (kernel)","grad. ascent dual",  "klasyfikacja"],
    ["7", "QFTKernel+SVM",   "4",  "brak (kernel)","grad. ascent dual",  "klasyfikacja"],
    ["8", "QRC + Ridge",     "6",  "7 (ridge)",   "Gauss-Jordan",       "klasyfikacja"],
    ["9", "Barren Plateau",  "2–8","n²",          "brak (diagnostyka)", "analiza BP"],
]

col_widths = [0.5*cm, 3.8*cm, 1.4*cm, 2.4*cm, 3.1*cm, 3.0*cm]
t = Table(table_data, colWidths=col_widths)
t.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a237e")),
    ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
    ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("ALIGN",        (0,0), (-1,-1), "CENTER"),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
    ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#bdbdbd")),
    ("TOPPADDING",   (0,0), (-1,-1), 4),
    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
]))
story.append(t)
story.append(Spacer(1, 12))
story.append(Paragraph(
    "Wszystkie algorytmy kwantowe używają wspólnej warstwy niskopoziomowej: "
    "<font face='Courier'>QuantumState</font> (statevector 2ⁿ amplitud), "
    "<font face='Courier'>QuantumGate</font> (fabryki X/Y/Z/H/RZ/RY/CNOT) i "
    "<font face='Courier'>QuantumCircuit</font> (sekwencja operacji z parametrami). "
    "Miary metryczne: <font face='Courier'>expectation_z()</font> jako obserwabla wyjściowa we wszystkich wariantach.",
    body_style
))

doc.build(story)
print("PDF zapisany:", OUTPUT)
