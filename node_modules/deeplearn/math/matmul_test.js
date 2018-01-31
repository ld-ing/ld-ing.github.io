"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var matmul_1 = require("./backends/types/matmul");
var ndarray_1 = require("./ndarray");
var commonTests = function (it) {
    it('A x B', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([3, 2], [0, 1, -3, 2, 2, 1]);
        var c = math.matMul(a, b);
        expect(c.shape).toEqual([2, 2]);
        test_util.expectArraysClose(c, [0, 8, -3, 20]);
    });
    it('A x B^t', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
        var c = math.matMul(a, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED);
        var expected = [7, 10, 16, 31];
        test_util.expectArraysClose(c, expected);
    });
    it('A^t x B', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
        var c = math.matMul(a, b, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR);
        var expected = [17, 12, 2, 22, 15, 4, 27, 18, 6];
        test_util.expectArraysClose(c, expected);
    });
    it('A^t x B^t', function (math) {
        var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
        var c = math.matMul(a, b, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.TRANSPOSED);
        var expected = [11, 13, 14, 20];
        test_util.expectArraysClose(c, expected);
    });
    it('A x B^t shapes do not match', function (math) {
        var a = dl.zeros([2, 3]);
        var b = dl.zeros([3, 2]);
        var f = function () {
            math.matMul(a, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED);
        };
        expect(f).toThrowError();
    });
    it('A^t x B shapes do not match', function (math) {
        var a = dl.zeros([2, 3]);
        var b = dl.zeros([3, 2]);
        var f = function () {
            math.matMul(a, b, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR);
        };
        expect(f).toThrowError();
    });
    it('A^t x B^t shapes do not match', function (math) {
        var a = dl.zeros([3, 2]);
        var b = dl.zeros([3, 2]);
        var f = function () {
            math.matMul(a, b, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.TRANSPOSED);
        };
        expect(f).toThrowError();
    });
    it('matmul throws when inner dimensions dont match', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
        expect(function () { return math.matMul(a, b); }).toThrowError();
    });
    it('matmul throws when passed non matrices', function (math) {
        var a = ndarray_1.Array3D.new([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        var b = ndarray_1.Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
        expect(function () { return math.matMul(a, b); }).toThrowError();
        expect(function () { return math.matMul(b, a); }).toThrowError();
    });
    it('Vector times matrix', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var result = math.vectorTimesMatrix(v, matrix);
        var expected = [11, 16];
        test_util.expectArraysClose(result, expected);
    });
    it('Vector times matrix with implicit reshape', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var result = math.vectorTimesMatrix(v, matrix);
        var expected = [11, 16];
        test_util.expectArraysClose(result, expected);
    });
    it('Vector times matrix throws when not passed a vector', function (math) {
        var v = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(function () { return math.vectorTimesMatrix(v, matrix); }).toThrowError();
    });
    it('Vector times matrix throws when not passed a matrix', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
        expect(function () { return math.vectorTimesMatrix(v, matrix); }).toThrowError();
    });
    it('Matrix times vector', function (math) {
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var v = ndarray_1.Array1D.new([2, 3]);
        var result = math.matrixTimesVector(matrix, v);
        var expected = [8, 18];
        test_util.expectArraysClose(result, expected);
    });
    it('Matrix * vector propagates NaNs', function (math) {
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var v = ndarray_1.Array1D.new([2, NaN]);
        var result = math.matrixTimesVector(matrix, v);
        var expected = [NaN, NaN];
        test_util.expectArraysClose(result, expected);
    });
    it('matrix times vector throws when not passed a vector', function (math) {
        var v = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(function () { return math.matrixTimesVector(matrix, v); }).toThrowError();
    });
    it('matrix times vector throws when not passed a matrix', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
        expect(function () { return math.matrixTimesVector(matrix, v); }).toThrowError();
    });
    it('Dot product', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        var result = math.dotProduct(v1, v2);
        test_util.expectNumbersClose(result.get(), 7);
    });
    it('Dot product propagates NaNs', function (math) {
        var v1 = ndarray_1.Array1D.new([2, NaN]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        var result = math.dotProduct(v1, v2);
        expect(result.get()).toEqual(NaN);
    });
    it('Dot product throws when vectors are different size', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        expect(function () { return math.dotProduct(v1, v2); }).toThrowError();
        expect(function () { return math.dotProduct(v2, v1); }).toThrowError();
    });
    it('Dot product throws when passed non vectors', function (math) {
        var v1 = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        expect(function () { return math.dotProduct(v1, v2); }).toThrowError();
        expect(function () { return math.dotProduct(v2, v1); }).toThrowError();
    });
    it('Outer product', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        var result = math.outerProduct(v1, v2);
        var expected = [4, 2, 6, 3];
        expect(result.shape).toEqual([2, 2]);
        test_util.expectArraysClose(result, expected);
    });
    it('gradients: A * B', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 10, 20, 30]);
        var b = ndarray_1.Array2D.new([3, 2], [2, 3, 4, 1, 2, 3]);
        var dy = ndarray_1.Array2D.new([2, 2], [1, 10, 20, 30]);
        var gradients = math.vjp(function () { return math.matMul(a, b, matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.REGULAR); }, { a: a, b: b }, dy);
        test_util.expectArraysClose(gradients.a, [
            dy.get(0, 0) * b.get(0, 0) + dy.get(0, 1) * b.get(0, 1),
            dy.get(0, 0) * b.get(1, 0) + dy.get(0, 1) * b.get(1, 1),
            dy.get(0, 0) * b.get(2, 0) + dy.get(0, 1) * b.get(2, 1),
            dy.get(1, 0) * b.get(0, 0) + dy.get(1, 1) * b.get(0, 1),
            dy.get(1, 0) * b.get(1, 0) + dy.get(1, 1) * b.get(1, 1),
            dy.get(1, 0) * b.get(2, 0) + dy.get(1, 1) * b.get(2, 1)
        ], 1e-1);
        expect(gradients.b.shape).toEqual(b.shape);
        test_util.expectArraysClose(gradients.b, [
            a.get(0, 0) * dy.get(0, 0) + a.get(1, 0) * dy.get(1, 0),
            a.get(0, 0) * dy.get(0, 1) + a.get(1, 0) * dy.get(1, 1),
            a.get(0, 1) * dy.get(0, 0) + a.get(1, 1) * dy.get(1, 0),
            a.get(0, 1) * dy.get(0, 1) + a.get(1, 1) * dy.get(1, 1),
            a.get(0, 2) * dy.get(0, 0) + a.get(1, 2) * dy.get(1, 0),
            a.get(0, 2) * dy.get(0, 1) + a.get(1, 2) * dy.get(1, 1)
        ], 1e-1);
    });
};
var gpuTests = function (it) {
    it('Matrix times vector, large matrix', function (math) {
        var maxTexSize = 16000;
        var sharedDim = maxTexSize + 4;
        var matrix = dl.zeros([2, sharedDim]);
        matrix.set(1, 0, sharedDim - 3);
        matrix.set(1, 0, sharedDim - 2);
        var v = dl.zeros([sharedDim]);
        v.set(1, sharedDim - 3);
        v.set(1, sharedDim - 2);
        var result = math.matrixTimesVector(matrix, v);
        var expected = [2, 0];
        test_util.expectArraysClose(result, expected);
    });
};
test_util.describeMathCPU('matMul', [commonTests]);
test_util.describeMathGPU('matMul', [commonTests, gpuTests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=matmul_test.js.map