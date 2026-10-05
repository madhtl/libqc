cmake_minimum_required(VERSION 3.31)
project(libqc)

set(CMAKE_CXX_STANDARD 20)

add_library(libqc SHARED
    src/library.cpp
    src/Qubit.cpp
    src/Matrix.cpp
    src/QuantumGate.cpp
    src/QuantumState.cpp
    src/QuantumCircuit.cpp
    src/MakeMoons.cpp
    src/QNN.cpp
    src/MLP.cpp
    src/QuantumKernel.cpp
    src/SVM.cpp
    src/VQE.cpp
    src/BarrenPlateau.cpp
    src/PEC.cpp
)

target_include_directories(libqc PUBLIC include)

add_executable(test_libqc tests/main.cpp)
target_link_libraries(test_libqc PRIVATE libqc)

add_executable(experiment src/main_experiment.cpp)
target_link_libraries(experiment PRIVATE libqc)

add_executable(experiment2 src/main_experiment2.cpp)
target_link_libraries(experiment2 PRIVATE libqc)

add_executable(experiment3 src/main_experiment3.cpp)
target_link_libraries(experiment3 PRIVATE libqc)

add_executable(experiment4 src/main_experiment4.cpp)
target_link_libraries(experiment4 PRIVATE libqc)

add_executable(experiment5 src/main_experiment5.cpp)
target_link_libraries(experiment5 PRIVATE libqc)

add_executable(experiment6
    src/main_experiment6.cpp
    src/QNN_Reupload.cpp
)
target_link_libraries(experiment6 PRIVATE libqc)

add_executable(experiment7
    src/main_experiment7.cpp
    src/QuantumReservoir.cpp
)
target_link_libraries(experiment7 PRIVATE libqc)

add_executable(experiment8
    src/main_experiment8.cpp
    src/QFTKernel.cpp
)
target_link_libraries(experiment8 PRIVATE libqc)

add_executable(experiment9
    src/main_experiment9.cpp
    src/NoiseModel.cpp
)
target_link_libraries(experiment9 PRIVATE libqc)
