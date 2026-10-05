//
// Created by Madzia Hatala on 24/09/2025.
//

#ifndef LIBQC_MATRIX_H
#define LIBQC_MATRIX_H
#include <vector>
#include <complex>
typedef std::complex<double> Complex;
class Matrix{
private:
    size_t rows, columns;
    std::vector<Complex> data; // 1d array of complex doubles
public:
    Matrix(size_t r,size_t c);

    Complex& operator()(size_t r, size_t c); //instead of member function
    const Complex& operator()(size_t r, size_t c) const;
    [[nodiscard]] size_t numRows() const{ return rows;}
    [[nodiscard]] size_t numCols() const{ return columns;}
    [[nodiscard]] Matrix Scalar(size_t m) const;
    [[nodiscard]]Matrix Transpose() const;
    static Complex dot(const std::vector<Complex>& row,const std::vector<Complex>& col);
    [[nodiscard]]std::vector<Complex> getRow(size_t row) const;
    [[nodiscard]]std::vector<Complex> getCol(size_t col) const;
    [[nodiscard]]Matrix Multiply(const Matrix& factor) const;
    [[nodiscard]]bool IsIdentity() const;
    [[nodiscard]]bool IsSquare() const{ return columns==rows;}
    [[nodiscard]]static Complex ConjNum(const Complex& complex);
    [[nodiscard]]Matrix ConjMatrix() const;
    [[nodiscard]]Matrix Conj_Transpose() const;
    [[nodiscard]]bool IsUnitary() const;

    void printAddress();
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);


};

#endif //LIBQC_MATRIX_H
