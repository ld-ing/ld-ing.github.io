"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('reverse a 1D array', function (math) {
            var input = ndarray_1.Array1D.new([1, 2, 3, 4, 5]);
            var result = math.reverse1D(input);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [5, 4, 3, 2, 1]);
        });
    };
    test_util.describeMathCPU('reverse1D', [tests]);
    test_util.describeMathGPU('reverse1D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('reverse a 2D array at axis [0]', function (math) {
            var axis = [0];
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var result = math.reverse2D(a, axis);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [4, 5, 6, 1, 2, 3]);
        });
        it('reverse a 2D array at axis [1]', function (math) {
            var axis = [1];
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var result = math.reverse2D(a, axis);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [3, 2, 1, 6, 5, 4]);
        });
        it('throws error with invalid input', function (math) {
            var x = ndarray_1.Array1D.new([1, 20, 300, 4]);
            expect(function () { return math.reverse2D(x, [0]); }).toThrowError();
        });
        it('throws error with invalid axis param', function (math) {
            var x = ndarray_1.Array2D.new([1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse2D(x, [2]); }).toThrowError();
            expect(function () { return math.reverse2D(x, [-3]); }).toThrowError();
        });
        it('throws error with non integer axis param', function (math) {
            var x = ndarray_1.Array2D.new([1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse2D(x, [0.5]); }).toThrowError();
        });
    };
    test_util.describeMathCPU('reverse2D', [tests]);
    test_util.describeMathGPU('reverse2D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        var shape = [2, 3, 4];
        var data = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        ];
        it('reverse a 3D array at axis [0]', function (math) {
            var input = ndarray_1.Array3D.new(shape, data);
            var result = math.reverse3D(input, [0]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            ]);
        });
        it('reverse a 3D array at axis [1]', function (math) {
            var input = ndarray_1.Array3D.new(shape, data);
            var result = math.reverse3D(input, [1]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15
            ]);
        });
        it('reverse a 3D array at axis [2]', function (math) {
            var input = ndarray_1.Array3D.new(shape, data);
            var result = math.reverse3D(input, [2]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8,
                15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20
            ]);
        });
        it('reverse a 3D array at axis [0, 1]', function (math) {
            var input = ndarray_1.Array3D.new(shape, data);
            var result = math.reverse3D(input, [0, 1]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15,
                8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
            ]);
        });
        it('reverse a 3D array at axis [0, 2]', function (math) {
            var input = ndarray_1.Array3D.new(shape, data);
            var result = math.reverse3D(input, [0, 2]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8
            ]);
        });
        it('reverse a 3D array at axis [1, 2]', function (math) {
            var input = ndarray_1.Array3D.new(shape, data);
            var result = math.reverse3D(input, [1, 2]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12
            ]);
        });
        it('throws error with invalid input', function (math) {
            var x = ndarray_1.Array2D.new([1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse3D(x, [1]); }).toThrowError();
        });
        it('throws error with invalid axis param', function (math) {
            var x = ndarray_1.Array3D.new([1, 1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse3D(x, [3]); }).toThrowError();
            expect(function () { return math.reverse3D(x, [-4]); }).toThrowError();
        });
        it('throws error with non integer axis param', function (math) {
            var x = ndarray_1.Array3D.new([1, 1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse3D(x, [0.5]); }).toThrowError();
        });
    };
    test_util.describeMathCPU('reverse3D', [tests]);
    test_util.describeMathGPU('reverse3D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        var shape = [3, 2, 3, 4];
        var data = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71
        ];
        it('reverse a 4D array at axis [0]', function (math) {
            var input = ndarray_1.Array4D.new(shape, data);
            var result = math.reverse4D(input, [0]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                63, 64, 65, 66, 67, 68, 69, 70, 71, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
            ]);
        });
        it('reverse a 4D array at axis [1]', function (math) {
            var input = ndarray_1.Array4D.new(shape, data);
            var result = math.reverse4D(input, [1]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2,
                3, 4, 5, 6, 7, 8, 9, 10, 11, 36, 37, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
            ]);
        });
        it('reverse a 4D array at axis [2]', function (math) {
            var input = ndarray_1.Array4D.new(shape, data);
            var result = math.reverse4D(input, [2]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 20, 21, 22,
                23, 16, 17, 18, 19, 12, 13, 14, 15, 32, 33, 34, 35, 28, 29,
                30, 31, 24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36,
                37, 38, 39, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51,
                68, 69, 70, 71, 64, 65, 66, 67, 60, 61, 62, 63
            ]);
        });
        it('reverse a 4D array at axis [3]', function (math) {
            var input = ndarray_1.Array4D.new(shape, data);
            var result = math.reverse4D(input, [3]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13,
                12, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30,
                29, 28, 35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47,
                46, 45, 44, 51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56,
                63, 62, 61, 60, 67, 66, 65, 64, 71, 70, 69, 68
            ]);
        });
        it('reverse a 4D array at axis [0, 2]', function (math) {
            var input = ndarray_1.Array4D.new(shape, data);
            var result = math.reverse4D(input, [0, 2]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 68, 69, 70,
                71, 64, 65, 66, 67, 60, 61, 62, 63, 32, 33, 34, 35, 28, 29,
                30, 31, 24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36,
                37, 38, 39, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15
            ]);
        });
        it('reverse a 4D array at axis [1, 3]', function (math) {
            var input = ndarray_1.Array4D.new(shape, data);
            var result = math.reverse4D(input, [1, 3]);
            expect(result.shape).toEqual(input.shape);
            test_util.expectArraysClose(result, [
                15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 3, 2, 1,
                0, 7, 6, 5, 4, 11, 10, 9, 8, 39, 38, 37, 36, 43, 42,
                41, 40, 47, 46, 45, 44, 27, 26, 25, 24, 31, 30, 29, 28, 35,
                34, 33, 32, 63, 62, 61, 60, 67, 66, 65, 64, 71, 70, 69, 68,
                51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56
            ]);
        });
        it('throws error with invalid input', function (math) {
            var x = ndarray_1.Array3D.new([1, 1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse4D(x, [1]); }).toThrowError();
        });
        it('throws error with invalid axis param', function (math) {
            var x = ndarray_1.Array4D.new([1, 1, 1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse4D(x, [4]); }).toThrowError();
            expect(function () { return math.reverse4D(x, [-5]); }).toThrowError();
        });
        it('throws error with non integer axis param', function (math) {
            var x = ndarray_1.Array4D.new([1, 1, 1, 4], [1, 20, 300, 4]);
            expect(function () { return math.reverse4D(x, [0.5]); }).toThrowError();
        });
    };
    test_util.describeMathCPU('reverse4D', [tests]);
    test_util.describeMathGPU('reverse4D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=reverse_test.js.map