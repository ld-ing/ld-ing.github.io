"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('divide', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 2, 5]);
            var r = math.divide(a, c);
            test_util.expectArraysClose(r, [1, 1, 1, 1, 2.5, 6 / 5]);
        });
        it('divide propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [1, 2]);
            var c = ndarray_1.Array2D.new([2, 1], [3, NaN]);
            var r = math.divide(a, c);
            test_util.expectArraysClose(r, [1 / 3, NaN]);
        });
        it('divide broadcasting same rank NDArrays different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 1], [2, 3]);
            var result = math.divide(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [1 / 2, 1, -1, -4 / 3];
            test_util.expectArraysClose(result, expected);
        });
        it('divide broadcast 2D + 1D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array1D.new([1, 2]);
            var result = math.divide(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [1, 1, -3, -2];
            test_util.expectArraysClose(result, expected);
        });
        it('div throws when passed ndarrays of different shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            expect(function () { return math.divide(a, b); }).toThrowError();
            expect(function () { return math.divide(b, a); }).toThrowError();
        });
        it('scalar divided by array', function (math) {
            var c = ndarray_1.Scalar.new(2);
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var r = math.divide(c, a);
            test_util.expectArraysClose(r, [2 / 1, 2 / 2, 2 / 3, 2 / 4, 2 / 5, 2 / 6]);
        });
        it('scalar divided by array propagates NaNs', function (math) {
            var c = ndarray_1.Scalar.new(NaN);
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 3]);
            var r = math.divide(c, a);
            test_util.expectArraysEqual(r, [NaN, NaN, NaN]);
        });
        it('array divided by scalar', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c = ndarray_1.Scalar.new(2);
            var r = math.divide(a, c);
            test_util.expectArraysClose(r, [1 / 2, 2 / 2, 3 / 2, 4 / 2, 5 / 2, 6 / 2]);
        });
        it('array divided by scalar propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, NaN]);
            var c = ndarray_1.Scalar.new(2);
            var r = math.divide(a, c);
            test_util.expectArraysClose(r, [1 / 2, 2 / 2, NaN]);
        });
        it('gradient: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var b = ndarray_1.Scalar.new(2);
            var dy = ndarray_1.Scalar.new(4);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [4 / 2]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-4 * 5 / (2 * 2)]);
        });
        it('gradient: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([3, 4, 5]);
            var dy = ndarray_1.Array1D.new([1, 10, 20]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [1 / 3, 10 / 4, 20 / 5]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-1 * 1 / 9, -10 * 2 / 16, -20 * 3 / 25]);
        });
        it('gradient: Array1D with int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var b = ndarray_1.Array1D.new([3, 4, 5], 'int32');
            var dy = ndarray_1.Array1D.new([1, 10, 20]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [1 / 3, 10 / 4, 20 / 5]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-1 * 1 / 9, -10 * 2 / 16, -20 * 3 / 25]);
        });
        it('gradient: 1d<int32> with 1d<bool> ', function (math) {
            var a = ndarray_1.Array1D.new([true, false, true], 'bool');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var dy = ndarray_1.Array1D.new([1, 19, 20]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [1, 19 / 2, 20 / 3]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-1 / 1, 0, -20 / 9]);
        });
        it('gradient: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [3, 1, 2, 3]);
            var b = ndarray_1.Array2D.new([2, 2], [1, 3, 4, 5]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 10, 15, 20]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [1 / 1, 10 / 3, 15 / 4, 20 / 5], 1e-1);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-1 * 3 / 1, -10 * 1 / 9, -15 * 2 / 16, -20 * 3 / 25], 1e-1);
        });
        it('gradient: scalar / Array1D', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([3, 4, 5]);
            var dy = ndarray_1.Array1D.new([6, 7, 8]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [6 / 3 + 7 / 4 + 8 / 5]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-6 * 2 / 9, -7 * 2 / 16, -8 * 2 / 25]);
        });
        it('gradient: Array2D / scalar', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[2, 3], [4, 5]]);
            var b = ndarray_1.Scalar.new(2);
            var dy = ndarray_1.Array2D.new([2, 2], [[6, 7], [8, 9]]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [6 / 2, 7 / 2, 8 / 2, 9 / 2], 1e-1);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-6 * 2 / 4 + -7 * 3 / 4 + -8 * 4 / 4 + -9 * 5 / 4], 1e-1);
        });
        it('gradient: Array2D / Array2D w/ broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [3, 4]);
            var b = ndarray_1.Array2D.new([2, 2], [[2, 3], [4, 5]]);
            var dy = ndarray_1.Array2D.new([2, 2], [[6, 7], [8, 9]]);
            var vjp = math.vjp(function () { return math.divide(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [6 / 2 + 7 / 3, 8 / 4 + 9 / 5], 1e-1);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [-6 * 3 / 4, -7 * 3 / 9, -8 * 4 / 16, -9 * 4 / 25], 1e-1);
        });
    };
    test_util.describeMathCPU('divide', [tests]);
    test_util.describeMathGPU('divide', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('multiplyStrict same-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            var expected = [5, 6, -12, 28];
            var result = math.multiplyStrict(a, b);
            expect(result.shape).toEqual([2, 2]);
            expect(result.dtype).toBe('float32');
            test_util.expectArraysClose(result, expected);
        });
        it('multiplyStrict propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 3, 4, 0]);
            var b = ndarray_1.Array2D.new([2, 2], [NaN, 3, NaN, 3]);
            var result = math.multiplyStrict(a, b);
            expect(result.dtype).toBe('float32');
            test_util.expectArraysClose(result, [NaN, 9, NaN, 0]);
        });
        it('multiplyStrict throws when passed ndarrays of different shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            expect(function () { return math.multiplyStrict(a, b); }).toThrowError();
            expect(function () { return math.multiplyStrict(b, a); }).toThrowError();
        });
        it('multiplyStrict throws when dtypes do not match', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7], 'int32');
            expect(function () { return math.multiplyStrict(a, b); })
                .toThrowError();
            expect(function () { return math.multiplyStrict(b, a); })
                .toThrowError();
        });
        it('multiplyStrict int32 * int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4], 'int32');
            var b = ndarray_1.Array2D.new([2, 2], [2, 1, 3, -4], 'int32');
            var res = math.multiplyStrict(a, b);
            expect(res.dtype).toBe('int32');
            test_util.expectArraysClose(res, [2, 2, -9, 16]);
        });
        it('same-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            var expected = [5, 6, -12, 28];
            var result = math.multiply(a, b);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result, expected);
        });
        it('broadcasting ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Scalar.new(2);
            var expected = [2, 4, -6, -8];
            var result = math.multiply(a, b);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result, expected);
        });
        it('broadcasting same rank NDArrays different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 1], [2, 3]);
            var result = math.multiply(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [2, 4, -9, -12];
            test_util.expectArraysClose(result, expected);
        });
        it('broadcast 2D + 1D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array1D.new([1, 2]);
            var result = math.multiply(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [1, 4, -3, -8];
            test_util.expectArraysClose(result, expected);
        });
        it('gradient: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var b = ndarray_1.Scalar.new(2);
            var dy = ndarray_1.Scalar.new(4);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [b.get() * dy.get()]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [a.get() * dy.get()]);
        });
        it('gradient: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([3, 4, 5]);
            var dy = ndarray_1.Array1D.new([1, 10, 20]);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [3 * 1, 4 * 10, 5 * 20]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [1 * 1, 2 * 10, 3 * 20]);
        });
        it('gradient: Array1D with dtype int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var b = ndarray_1.Array1D.new([3, 4, 5], 'int32');
            var dy = ndarray_1.Array1D.new([1, 10, 20]);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [3 * 1, 4 * 10, 5 * 20]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [1 * 1, 2 * 10, 3 * 20]);
        });
        it('gradient: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [3, 1, 2, 3]);
            var b = ndarray_1.Array2D.new([2, 2], [1, 3, 4, 5]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 10, 15, 20]);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [1 * 1, 3 * 10, 4 * 15, 5 * 20], 1e-1);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [3 * 1, 1 * 10, 2 * 15, 3 * 20], 1e-1);
        });
        it('gradient: scalar * Array1D', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([3, 4, 5]);
            var dy = ndarray_1.Array1D.new([6, 7, 8]);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [3 * 6 + 4 * 7 + 5 * 8]);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [2 * 6, 2 * 7, 2 * 8]);
        });
        it('gradient: Array2D * scalar', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[2, 3], [4, 5]]);
            var b = ndarray_1.Scalar.new(2);
            var dy = ndarray_1.Array2D.new([2, 2], [[6, 7], [8, 9]]);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [2 * 6, 2 * 7, 2 * 8, 2 * 9], 1e-1);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [2 * 6 + 3 * 7 + 4 * 8 + 5 * 9], 1e-1);
        });
        it('gradient: Array2D * Array2D w/ broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [3, 4]);
            var b = ndarray_1.Array2D.new([2, 2], [[2, 3], [4, 5]]);
            var dy = ndarray_1.Array2D.new([2, 2], [[6, 7], [8, 9]]);
            var vjp = math.vjp(function () { return math.multiply(a, b); }, { a: a, b: b }, dy);
            expect(vjp.a.shape).toEqual(a.shape);
            expect(vjp.a.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.a, [2 * 6 + 3 * 7, 4 * 8 + 5 * 9], 1e-1);
            expect(vjp.b.shape).toEqual(b.shape);
            expect(vjp.b.dtype).toEqual('float32');
            test_util.expectArraysClose(vjp.b, [6 * 3, 7 * 3, 8 * 4, 9 * 4], 1e-1);
        });
    };
    test_util.describeMathCPU('multiply', [tests]);
    test_util.describeMathGPU('multiply', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('same-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, -2, -3, 0, 7, 1]);
            var b = ndarray_1.Array2D.new([2, 3], [5, 3, 4, 5, 2, -3], 'int32');
            var expected = [1, -8, 81, 0, 49, 1];
            var result = math.pow(a, b);
            expect(result.shape).toEqual([2, 3]);
            test_util.expectArraysClose(result, expected, 0.01);
        });
        it('int32^int32 returns int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var exp = ndarray_1.Scalar.new(2, 'int32');
            var result = math.pow(a, exp);
            expect(result.shape).toEqual([3]);
            expect(result.dtype).toBe('int32');
            test_util.expectArraysEqual(result, [1, 4, 9]);
        });
        it('different-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, -2, -3, 0, 7, 1]);
            var b = ndarray_1.Scalar.new(2, 'int32');
            var expected = [1, 4, 9, 0, 49, 1];
            var result = math.pow(a, b);
            expect(result.shape).toEqual([2, 3]);
            test_util.expectArraysClose(result, expected, 0.05);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [NaN, 3, NaN, 0]);
            var b = ndarray_1.Array2D.new([2, 2], [1, 3, 2, 3], 'int32');
            var result = math.pow(a, b);
            test_util.expectArraysClose(result, [NaN, 27, NaN, 0], 0.05);
        });
        it('throws when passed non int32 exponent param', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7], 'float32');
            expect(function () { return math.pow(a, b); }).toThrowError();
        });
        it('broadcasting same rank NDArrays different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 1], [2, 1], 'int32');
            var result = math.pow(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [1, 4, -3, -4];
            test_util.expectArraysClose(result, expected);
        });
        it('broadcast 2D + 1D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array1D.new([1, 2], 'int32');
            var result = math.pow(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [1, 4, -3, 16];
            test_util.expectArraysClose(result, expected);
        });
        it('powStrict same-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, -2, -3, 0, 7, 1]);
            var b = ndarray_1.Array2D.new([2, 3], [5, 3, 4, 5, 2, -3], 'int32');
            var expected = [1, -8, 81, 0, 49, 1];
            var result = math.powStrict(a, b);
            expect(result.shape).toEqual([2, 3]);
            test_util.expectArraysClose(result, expected, 0.01);
        });
        it('powStrict throws when passed ndarrays of different shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7], 'int32');
            expect(function () { return math.powStrict(a, b); }).toThrowError();
        });
        it('powStrict throws when passed non int32 exponent param', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7], 'float32');
            expect(function () { return math.powStrict(a, b); }).toThrowError();
        });
        it('gradients: Scalar ^ Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var b = ndarray_1.Scalar.new(2, 'int32');
            var dy = ndarray_1.Scalar.new(3);
            var gradients = math.vjp(function () { return math.pow(a, b); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [2 * 5 * 3], 1e-1);
        });
        it('gradients: NDArray ^ NDArray', function (math) {
            var a = ndarray_1.Array1D.new([-1, .5, 2]);
            var b = ndarray_1.Array1D.new([3, 2, -1], 'int32');
            var dy = ndarray_1.Array1D.new([1, 5, 10]);
            var gradients = math.vjp(function () { return math.pow(a, b); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                3 * Math.pow(-1, 2) * 1, 2 * Math.pow(.5, 1) * 5,
                -1 * Math.pow(2, -2) * 10
            ], 1e-1);
        });
    };
    test_util.describeMathCPU('pow', [tests]);
    test_util.describeMathGPU('pow', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('c + A', function (math) {
            var c = ndarray_1.Scalar.new(5);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var result = math.add(c, a);
            test_util.expectArraysClose(result, [6, 7, 8]);
        });
        it('c + A propagates NaNs', function (math) {
            var c = ndarray_1.Scalar.new(NaN);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var res = math.add(c, a);
            test_util.expectArraysEqual(res, [NaN, NaN, NaN]);
        });
        it('A + B broadcasting same rank NDArrays different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 1], [2, 3]);
            var result = math.add(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [3, 4, 0, -1];
            test_util.expectArraysClose(result, expected);
        });
        it('A + B broadcast 2D + 1D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array1D.new([1, 2]);
            var result = math.add(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [2, 4, -2, -2];
            test_util.expectArraysClose(result, expected);
        });
        it('A + B', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            var result = math.add(a, b);
            var expected = [6, 7, 0];
            test_util.expectArraysClose(result, expected);
        });
        it('A + B propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, NaN]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            var res = math.add(a, b);
            test_util.expectArraysClose(res, [6, 7, NaN]);
        });
        it('A + B throws when passed ndarrays with different shape', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1, 5]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.add(a, b); }).toThrowError();
            expect(function () { return math.add(b, a); }).toThrowError();
        });
        it('2D+scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res, [3, 4, 5, 6, 7, 8]);
        });
        it('scalar+1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 6]);
            var res = math.add(a, b);
            expect(res.shape).toEqual([6]);
            test_util.expectArraysClose(res, [3, 4, 5, 6, 7, 8]);
        });
        it('2D+2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res, [8, 9, 12, 4, 5, 8]);
        });
        it('2D+2D broadcast inner dim of b', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 5, 4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res, [8, 9, 12, 7, 8, 9]);
        });
        it('3D+scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3, 1]);
            test_util.expectArraysClose(res, [0, 1, 2, 3, 4, 5]);
        });
        it('gradient: scalar + 1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([3, 4, 5]);
            var dy = ndarray_1.Array1D.new([7, 8, 9]);
            var gradients = math.vjp(function () { return math.add(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [7 + 8 + 9], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [7, 8, 9], 1e-1);
        });
        it('gradient: 2D + 2D broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [2, 3]);
            var b = ndarray_1.Array2D.new([2, 2], [4, 5, 6, 7]);
            var dy = ndarray_1.Array2D.new([2, 2], [5, 4, 3, 2]);
            var gradients = math.vjp(function () { return math.add(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [5 + 4, 3 + 2], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [5, 4, 3, 2], 1e-1);
        });
    };
    test_util.describeMathCPU('add', [tests]);
    test_util.describeMathGPU('add', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('c - A', function (math) {
            var c = ndarray_1.Scalar.new(5);
            var a = ndarray_1.Array1D.new([7, 2, 3]);
            var result = math.subtract(c, a);
            test_util.expectArraysClose(result, [-2, 3, 2]);
        });
        it('A - c', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3]);
            var c = ndarray_1.Scalar.new(5);
            var result = math.subtract(a, c);
            test_util.expectArraysClose(result, [-4, -3, -8]);
        });
        it('A - c propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 3]);
            var c = ndarray_1.Scalar.new(5);
            var res = math.subtract(a, c);
            test_util.expectArraysClose(res, [-4, NaN, -2]);
        });
        it('A - B', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            var result = math.subtract(a, b);
            var expected = [-2, 3, 2];
            test_util.expectArraysClose(result, expected);
        });
        it('A - B propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1]);
            var b = ndarray_1.Array1D.new([4, NaN, -1]);
            var res = math.subtract(a, b);
            test_util.expectArraysClose(res, [-2, NaN, 2]);
        });
        it('A - B throws when passed ndarrays with different shape', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1, 5]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.subtract(a, b); }).toThrowError();
            expect(function () { return math.subtract(b, a); }).toThrowError();
        });
        it('A - B broadcasting same rank NDArrays different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 1], [2, 3]);
            var result = math.subtract(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [-1, 0, -6, -7];
            test_util.expectArraysClose(result, expected);
        });
        it('A - B broadcast 2D + 1D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array1D.new([1, 2]);
            var result = math.subtract(a, b);
            expect(result.shape).toEqual([2, 2]);
            var expected = [0, 0, -4, -6];
            test_util.expectArraysClose(result, expected);
        });
        it('2D-scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res, [-1, 0, 1, 2, 3, 4]);
        });
        it('scalar-1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 6]);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([6]);
            test_util.expectArraysClose(res, [1, 0, -1, -2, -3, -4]);
        });
        it('2D-2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res, [-6, -5, -2, -2, -1, 2]);
        });
        it('2D-2D broadcast inner dim of b', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 5, 4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res, [-6, -5, -2, 1, 2, 3]);
        });
        it('3D-scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3, 1]);
            test_util.expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
        });
        it('gradients: basic 1D arrays', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([3, 2, 1]);
            var dy = ndarray_1.Array1D.new([1, 10, 20]);
            var gradients = math.vjp(function () { return math.subtract(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [1, 10, 20], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [-1, -10, -20], 1e-1);
        });
        it('gradients: basic 2D arrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [0, 1, 2, 3]);
            var b = ndarray_1.Array2D.new([2, 2], [3, 2, 1, 0]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 10, 15, 20]);
            var gradients = math.vjp(function () { return math.subtract(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [1, 10, 15, 20], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [-1, -10, -15, -20], 1e-1);
        });
        it('gradient: 1D - scalar broadcast', function (math) {
            var a = ndarray_1.Array1D.new([3, 4, 5]);
            var b = ndarray_1.Scalar.new(2);
            var dy = ndarray_1.Array1D.new([7, 8, 9]);
            var gradients = math.vjp(function () { return math.subtract(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [7, 8, 9], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [-7 - 8 - 9], 1e-1);
        });
        it('gradient: scalar - 1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([3, 4, 5]);
            var dy = ndarray_1.Array1D.new([7, 8, 9]);
            var gradients = math.vjp(function () { return math.subtract(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [7 + 8 + 9], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [-7, -8, -9], 1e-1);
        });
        it('gradient: 2D - 2D broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [4, 5, 6, 7]);
            var b = ndarray_1.Array2D.new([2, 1], [2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [5, 4, 3, 2]);
            var gradients = math.vjp(function () { return math.subtract(a, b); }, { a: a, b: b }, dy);
            expect(gradients.a.shape).toEqual(a.shape);
            expect(gradients.a.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.a, [5, 4, 3, 2], 1e-1);
            expect(gradients.b.shape).toEqual(b.shape);
            expect(gradients.b.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients.b, [-5 - 4, -3 - 2], 1e-1);
        });
    };
    test_util.describeMathCPU('subtract', [tests]);
    test_util.describeMathGPU('subtract', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Scaled ndarray add', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
            var b = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c1 = ndarray_1.Scalar.new(3);
            var c2 = ndarray_1.Scalar.new(2);
            var result = math.scaledArrayAdd(c1, a, c2, b);
            expect(result.shape).toEqual([2, 3]);
            test_util.expectArraysClose(result, [8, 16, 24, 32, 40, 48]);
            var wrongSizeMat = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            expect(function () { return math.scaledArrayAdd(c1, wrongSizeMat, c2, b); })
                .toThrowError();
        });
        it('throws when passed non-scalars', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
            var b = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c1 = dl.randNormal([10]);
            var c2 = ndarray_1.Scalar.new(2);
            expect(function () { return math.scaledArrayAdd(c1, a, c2, b); }).toThrowError();
            expect(function () { return math.scaledArrayAdd(c2, a, c1, b); }).toThrowError();
        });
        it('throws when NDArrays are different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
            var b = ndarray_1.Array2D.new([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
            var c1 = ndarray_1.Scalar.new(3);
            var c2 = ndarray_1.Scalar.new(2);
            expect(function () { return math.scaledArrayAdd(c1, a, c2, b); }).toThrowError();
        });
    };
    test_util.describeMathCPU('scaledArrayAdd', [tests]);
    test_util.describeMathGPU('scaledArrayAdd', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=arithmetic_test.js.map