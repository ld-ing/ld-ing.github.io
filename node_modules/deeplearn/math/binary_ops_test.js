"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('basic', function (math) {
            var x = ndarray_1.Array1D.new([0, 1, -2, -4]);
            var a = ndarray_1.Array1D.new([0.15, 0.2, 0.25, 0.15]);
            var result = math.prelu(x, a);
            expect(result.shape).toEqual(x.shape);
            test_util.expectArraysClose(result, [0, 1, -0.5, -0.6]);
        });
        it('propagates NaN', function (math) {
            var x = ndarray_1.Array1D.new([0, 1, NaN]);
            var a = ndarray_1.Array1D.new([0.15, 0.2, 0.25]);
            var result = math.prelu(x, a);
            expect(result.shape).toEqual(x.shape);
            test_util.expectArraysClose(result, [0, 1, NaN]);
        });
        it('derivative', function (math) {
            var x = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var a = ndarray_1.Array1D.new([0.2, 0.4, 0.25, 0.15]);
            var dy = ndarray_1.Array1D.new([1, 1, 1, 1]);
            var dx = math.vjp(function () { return math.prelu(x, a); }, x, dy);
            expect(dx.shape).toEqual(x.shape);
            expect(dx.dtype).toEqual('float32');
            test_util.expectArraysClose(dx, [1, 1, 0.25, 0.15]);
        });
        it('derivative propagates NaN', function (math) {
            var x = ndarray_1.Array1D.new([0.5, -0.1, NaN]);
            var a = ndarray_1.Array1D.new([0.2, 0.3, 0.25]);
            var dy = ndarray_1.Array1D.new([5, 50, 500]);
            var dx = math.vjp(function () { return math.prelu(x, a); }, x, dy);
            expect(dx.shape).toEqual(x.shape);
            expect(dx.dtype).toEqual('float32');
            test_util.expectArraysClose(dx, [5, 50 * 0.3, NaN], 1e-1);
        });
    };
    test_util.describeMathCPU('prelu', [tests]);
    test_util.describeMathGPU('prelu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('float32 and float32', function (math) {
            var a = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var b = ndarray_1.Array1D.new([0.2, 0.4, 0.25, 0.15]);
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
        });
        it('int32 and int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 5, 2, 3], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 1, 4], 'int32');
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(a.shape);
            expect(result.dtype).toBe('int32');
            test_util.expectArraysEqual(result, [2, 5, 2, 4]);
        });
        it('bool and bool', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true], 'bool');
            var b = ndarray_1.Array1D.new([false, false, true, true], 'bool');
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(a.shape);
            expect(result.dtype).toBe('bool');
            test_util.expectArraysEqual(result, [true, false, true, true]);
        });
        it('different dtypes throws error', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true], 'float32');
            var b = ndarray_1.Array1D.new([false, false, true, true], 'int32');
            expect(function () { return math.maximum(a, b); }).toThrowError();
        });
        it('propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([0.5, -0.1, NaN]);
            var b = ndarray_1.Array1D.new([0.2, 0.3, 0.25]);
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0.5, 0.3, NaN]);
        });
        it('broadcasts array1d and scalar', function (math) {
            var a = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var b = ndarray_1.Scalar.new(0.6);
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
        });
        it('broadcasts scalar and array1d', function (math) {
            var a = ndarray_1.Scalar.new(0.6);
            var b = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(b.shape);
            test_util.expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
        });
        it('broadcasts array1d and array2d', function (math) {
            var a = ndarray_1.Array1D.new([0.5, 0.3]);
            var b = ndarray_1.Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(b.shape);
            test_util.expectArraysClose(result, [0.5, 0.4, 0.6, 0.3]);
        });
        it('broadcasts 2x1 array2d and 2x2 array2d', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [0.5, 0.3]);
            var b = ndarray_1.Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
            var result = math.maximum(a, b);
            expect(result.shape).toEqual(b.shape);
            test_util.expectArraysClose(result, [0.5, 0.5, 0.6, 0.3]);
        });
    };
    test_util.describeMathCPU('maximum', [tests]);
    test_util.describeMathGPU('maximum', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('float32 and float32', function (math) {
            var a = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var b = ndarray_1.Array1D.new([0.2, 0.4, 0.25, 0.15]);
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
        });
        it('int32 and int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 5, 2, 3], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 1, 4], 'int32');
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(a.shape);
            expect(result.dtype).toBe('int32');
            test_util.expectArraysEqual(result, [1, 3, 1, 3]);
        });
        it('bool and bool', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true], 'bool');
            var b = ndarray_1.Array1D.new([false, false, true, true], 'bool');
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(a.shape);
            expect(result.dtype).toBe('bool');
            test_util.expectArraysEqual(result, [false, false, false, true]);
        });
        it('different dtypes throws error', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true], 'float32');
            var b = ndarray_1.Array1D.new([false, false, true, true], 'int32');
            expect(function () { return math.minimum(a, b); }).toThrowError();
        });
        it('propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([0.5, -0.1, NaN]);
            var b = ndarray_1.Array1D.new([0.2, 0.3, 0.25]);
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0.2, -0.1, NaN]);
        });
        it('broadcasts array1d and scalar', function (math) {
            var a = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var b = ndarray_1.Scalar.new(0.6);
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
        });
        it('broadcasts scalar and array1d', function (math) {
            var a = ndarray_1.Scalar.new(0.6);
            var b = ndarray_1.Array1D.new([0.5, 3, -0.1, -4]);
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(b.shape);
            test_util.expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
        });
        it('broadcasts array1d and array2d', function (math) {
            var a = ndarray_1.Array1D.new([0.5, 0.3]);
            var b = ndarray_1.Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(b.shape);
            test_util.expectArraysClose(result, [0.2, 0.3, 0.5, 0.15]);
        });
        it('broadcasts 2x1 array2d and 2x2 array2d', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [0.5, 0.3]);
            var b = ndarray_1.Array2D.new([2, 2], [0.2, 0.4, 0.6, 0.15]);
            var result = math.minimum(a, b);
            expect(result.shape).toEqual(b.shape);
            test_util.expectArraysClose(result, [0.2, 0.4, 0.3, 0.15]);
        });
    };
    test_util.describeMathCPU('minimum', [tests]);
    test_util.describeMathGPU('minimum', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=binary_ops_test.js.map