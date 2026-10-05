//
// Created by Madzia Hatala on 01/10/2025.
//
#include "../include/Matrix.h"
#include <cassert>
#include <vector>
#include <iostream>
Matrix::Matrix(size_t r, size_t c) : rows(r), columns(c), data(r*c, Complex(0.0,0.0)){};

Complex& Matrix::operator()(size_t r, size_t c) { //returns alias- to make val actually modifiable
    return data[r*columns +c];
}
const Complex& Matrix::operator()(size_t r, size_t c) const{
    return data[r*columns +c];
}
Matrix Matrix::Scalar(size_t m) const {
    auto m_t = static_cast<double>(m);
    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result(i, j) =
                    (*this)(i, j) * m_t; //this= pointer, *this dereferences pointer to an actual object it points to
        }
    }
    return result;
}
bool Matrix::IsIdentity() const{
    if (columns!=rows){
        return false;
    }
    for (size_t i=0; i<rows; i++){
        for(size_t j=0; j<columns; j++){
            Complex expected = (i==j) ? Complex(1.0,0.0) : Complex(0.0,0.0);
            if(std::abs((*this)(i,j) - expected) > 1e-9){
                return false;
            }
        }
    }
    return true;
}
Matrix Matrix::Transpose() const {
    Matrix result(columns, rows);
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            result(i, j) =
                    (*this)(j, i);
        }
    }
    return result;


}
std::vector<Complex> Matrix::getRow(size_t row) const{
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    std::vector<Complex> result(columns);
    size_t j = 0;
    for(Complex &r : result)
    {
        r=(*this)(row,j);
                ++j;

    }
    return result;

}
std::vector<Complex> Matrix::getCol(size_t col) const{
    if (col >= columns) {
        throw std::out_of_range("Column index out of range");
    }
    std::vector<Complex> result(rows);
    size_t j = 0;
    for(Complex &r : result)
    {
        r=(*this)(j,col);
        ++j;

    }
    return result;

}
void Matrix::printAddress(){
    std::cout << this << std::endl;
}
Complex Matrix::dot(const std::vector<Complex>& row,const std::vector<Complex>& col) {
    Complex result(0.0,0.0);
    assert(row.size()==col.size());
    auto it1 = row.begin();
    auto it2 = col.begin();
    for(;it1!=row.end();it1++,it2++){
        result += *it1 * *it2;
    }
return result;
}


Matrix Matrix::Multiply(const Matrix& factor) const{
    Matrix result(rows, factor.columns);
    if (factor.IsIdentity()) {
        return *this;
    }
    if (this->IsIdentity()){
        return factor;
    }
    if(columns != factor.rows){
        throw std::invalid_argument("Matrices dimensions dont match.");
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < factor.columns; j++) {
            Complex sum = Complex(0.0,0.0);
            for (int k = 0; k<columns;k++) {
                sum += (*this)(i,k)*factor(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}
Complex Matrix::ConjNum(const Complex& complex){
    return Complex(complex.real(), -complex.imag());
}
Matrix Matrix::ConjMatrix()const{
    Matrix result(rows, columns);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            Complex const el_value = (*this)(i, j);
            result(i, j) =  ConjNum(el_value);
        }
    }
    return result;

}
Matrix Matrix::Conj_Transpose() const {
    return ConjMatrix().Transpose();
}
bool Matrix::IsUnitary() const {
    if (!IsSquare()) return false;
    return Conj_Transpose().Multiply(*this).IsIdentity();
}
std::ostream& operator<<(std::ostream& os, const Matrix& matrix){
    for(size_t i=0;i<matrix.rows; i++){
        for(size_t j=0;j<matrix.columns; j++){
            os << matrix(i,j) << " ";
        }
        os << "\n";
    }
    return os;
}
