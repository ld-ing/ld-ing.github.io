"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var reduce_util = require("./reduce_util");
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.min(a).get(), -7);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([3, NaN, 2]);
            expect(math.min(a).get()).toEqual(NaN);
        });
        it('2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.min(a).get(), -7);
        });
        it('2D axis=[0,1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.min(a, [0, 1]).get(), -7);
        });
        it('2D, axis=0', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            var r = math.min(a, 0);
            expect(r.shape).toEqual([3]);
            test_util.expectArraysClose(r, [3, -7, 0]);
        });
        it('2D, axis=0, keepDims', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            var r = math.min(a, 0, true);
            expect(r.shape).toEqual([1, 3]);
            test_util.expectArraysClose(r, [3, -7, 0]);
        });
        it('2D, axis=1 provided as a number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.min(a, 1);
            test_util.expectArraysClose(r, [2, -7]);
        });
        it('2D, axis = -1 provided as a number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.min(a, -1);
            test_util.expectArraysClose(r, [2, -7]);
        });
        it('2D, axis=[1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.min(a, [1]);
            test_util.expectArraysClose(r, [2, -7]);
        });
    };
    test_util.describeMathCPU('min', [tests]);
    test_util.describeMathGPU('min', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('with one element dominating', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2]);
            var r = math.max(a);
            test_util.expectNumbersClose(r.get(), 100);
        });
        it('with all elements being the same', function (math) {
            var a = ndarray_1.Array1D.new([3, 3, 3]);
            var r = math.max(a);
            test_util.expectNumbersClose(r.get(), 3);
        });
        it('propagates NaNs', function (math) {
            expect(math.max(ndarray_1.Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
        });
        it('2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.max(a).get(), 100);
        });
        it('2D axis=[0,1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.max(a, [0, 1]).get(), 100);
        });
        it('2D, axis=0', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            var r = math.max(a, [0]);
            expect(r.shape).toEqual([3]);
            test_util.expectArraysClose(r, [100, -1, 2]);
        });
        it('2D, axis=0, keepDims', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            var r = math.max(a, [0], true);
            expect(r.shape).toEqual([1, 3]);
            test_util.expectArraysClose(r, [100, -1, 2]);
        });
        it('2D, axis=1 provided as a number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.max(a, 1);
            test_util.expectArraysClose(r, [5, 100]);
        });
        it('2D, axis = -1 provided as a number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.max(a, -1);
            test_util.expectArraysClose(r, [5, 100]);
        });
        it('2D, axis=[1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.max(a, [1]);
            test_util.expectArraysClose(r, [5, 100]);
        });
    };
    test_util.describeMathCPU('max', [tests]);
    test_util.describeMathGPU('max', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 3, 2]);
            var result = math.argMax(a);
            expect(result.dtype).toBe('int32');
            expect(result.get()).toBe(2);
        });
        it('one value', function (math) {
            var a = ndarray_1.Array1D.new([10]);
            var result = math.argMax(a);
            expect(result.dtype).toBe('int32');
            expect(result.get()).toBe(0);
        });
        it('N > than parallelization threshold', function (math) {
            var n = reduce_util.PARALLELIZE_THRESHOLD * 2;
            var values = new Float32Array(n);
            for (var i = 0; i < n; i++) {
                values[i] = i;
            }
            var a = ndarray_1.Array1D.new(values);
            var result = math.argMax(a);
            expect(result.dtype).toBe('int32');
            expect(result.get()).toBe(n - 1);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, NaN, 3]);
            var res = math.argMax(a);
            expect(res.dtype).toBe('int32');
            test_util.assertIsNan(res.get(), res.dtype);
        });
        it('2D, no axis specified', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(math.argMax(a).get()).toBe(3);
        });
        it('2D, axis=0', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            var r = math.argMax(a, 0);
            expect(r.shape).toEqual([3]);
            expect(r.dtype).toBe('int32');
            test_util.expectArraysEqual(r, [1, 0, 1]);
        });
        it('2D, axis=1', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.argMax(a, 1);
            expect(r.dtype).toBe('int32');
            test_util.expectArraysEqual(r, [2, 0]);
        });
        it('2D, axis = -1', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.argMax(a, -1);
            expect(r.dtype).toBe('int32');
            test_util.expectArraysEqual(r, [2, 0]);
        });
    };
    test_util.describeMathCPU('argmax', [tests]);
    test_util.describeMathGPU('argmax', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 3, 2]);
            var result = math.argMin(a);
            expect(result.get()).toBe(1);
        });
        it('one value', function (math) {
            var a = ndarray_1.Array1D.new([10]);
            var result = math.argMin(a);
            expect(result.get()).toBe(0);
        });
        it('N > than parallelization threshold', function (math) {
            var n = reduce_util.PARALLELIZE_THRESHOLD * 2;
            var values = new Float32Array(n);
            for (var i = 0; i < n; i++) {
                values[i] = n - i;
            }
            var a = ndarray_1.Array1D.new(values);
            var result = math.argMin(a);
            expect(result.dtype).toBe('int32');
            expect(result.get()).toBe(n - 1);
        });
        it('Arg min propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, NaN, 7, 3]);
            var res = math.argMin(a);
            test_util.assertIsNan(res.get(), res.dtype);
        });
        it('2D, no axis specified', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(math.argMin(a).get()).toBe(4);
        });
        it('2D, axis=0', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            var r = math.argMin(a, 0);
            expect(r.shape).toEqual([3]);
            expect(r.dtype).toBe('int32');
            test_util.expectArraysEqual(r, [0, 1, 0]);
        });
        it('2D, axis=1', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, -8]);
            var r = math.argMin(a, 1);
            test_util.expectArraysEqual(r, [1, 2]);
        });
        it('2D, axis = -1', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, -8]);
            var r = math.argMin(a, -1);
            test_util.expectArraysEqual(r, [1, 2]);
        });
    };
    test_util.describeMathCPU('argmin', [tests]);
    test_util.describeMathGPU('argmin', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('equals', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, 7, 3]);
            var b = ndarray_1.Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
            var result = math.argMaxEquals(a, b);
            expect(result.get()).toBe(1);
        });
        it('not equals', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, 1, 3]);
            var b = ndarray_1.Array1D.new([-100.3, -20.0, -10.0, -5, 0]);
            var result = math.argMaxEquals(a, b);
            expect(result.get()).toBe(0);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([0, 3, 1, 3]);
            var b = ndarray_1.Array1D.new([NaN, -20.0, -10.0, -5]);
            var result = math.argMaxEquals(a, b);
            test_util.assertIsNan(result.get(), result.dtype);
        });
        it('throws when given arrays of different shape', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, 7, 3, 10]);
            var b = ndarray_1.Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
            expect(function () { return math.argMaxEquals(a, b); }).toThrowError();
        });
    };
    test_util.describeMathCPU('argMaxEquals', [tests]);
    test_util.describeMathGPU('argMaxEquals', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('0', function (math) {
            var a = ndarray_1.Scalar.new(0);
            var result = math.logSumExp(a);
            test_util.expectNumbersClose(result.get(), 0);
        });
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3]);
            var result = math.logSumExp(a);
            test_util.expectNumbersClose(result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, NaN]);
            var result = math.logSumExp(a);
            expect(result.get()).toEqual(NaN);
        });
        it('axes=0 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var r = math.logSumExp(a, [0]);
            expect(r.shape).toEqual([2]);
            var expected = [
                Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
                Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
            ];
            test_util.expectArraysClose(r, expected);
        });
        it('axes=0 in 2D array, keepDims', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var r = math.logSumExp(a, [0], true);
            expect(r.shape).toEqual([1, 2]);
            var expected = [
                Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
                Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
            ];
            test_util.expectArraysClose(r, expected);
        });
        it('axes=1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, [1]);
            expect(res.shape).toEqual([3]);
            var expected = [
                Math.log(Math.exp(1) + Math.exp(2)),
                Math.log(Math.exp(3) + Math.exp(0)),
                Math.log(Math.exp(0) + Math.exp(1)),
            ];
            test_util.expectArraysClose(res, expected);
        });
        it('axes = -1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, -1);
            expect(res.shape).toEqual([3]);
            var expected = [
                Math.log(Math.exp(1) + Math.exp(2)),
                Math.log(Math.exp(3) + Math.exp(0)),
                Math.log(Math.exp(0) + Math.exp(1)),
            ];
            test_util.expectArraysClose(res, expected);
        });
        it('2D, axes=1 provided as a single digit', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, 1);
            expect(res.shape).toEqual([2]);
            var expected = [
                Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
                Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
            ];
            test_util.expectArraysClose(res, expected);
        });
        it('axes=0,1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, [0, 1]);
            expect(res.shape).toEqual([]);
            var expected = [Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
                    Math.exp(1))];
            test_util.expectArraysClose(res, expected);
        });
    };
    test_util.describeMathCPU('logSumExp', [tests]);
    test_util.describeMathGPU('logSumExp', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var result = math.sum(a);
            test_util.expectNumbersClose(result.get(), 7);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
            expect(math.sum(a).get()).toEqual(NaN);
        });
        it('sum over dtype int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 5, 7, 3], 'int32');
            var sum = math.sum(a);
            expect(sum.get()).toBe(16);
        });
        it('sum over dtype bool', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true, true], 'bool');
            var sum = math.sum(a);
            expect(sum.get()).toBe(3);
        });
        it('sums all values in 2D array with keep dim', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, null, true);
            expect(res.shape).toEqual([1, 1]);
            test_util.expectArraysClose(res, [7]);
        });
        it('sums across axis=0 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [0]);
            expect(res.shape).toEqual([2]);
            test_util.expectArraysClose(res, [4, 3]);
        });
        it('sums across axis=0 in 2D array, keepDims', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [0], true);
            expect(res.shape).toEqual([1, 2]);
            test_util.expectArraysClose(res, [4, 3]);
        });
        it('sums across axis=1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [1]);
            expect(res.shape).toEqual([3]);
            test_util.expectArraysClose(res, [3, 3, 1]);
        });
        it('2D, axis=1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, 1);
            expect(res.shape).toEqual([2]);
            test_util.expectArraysClose(res, [6, 1]);
        });
        it('2D, axis = -1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, -1);
            expect(res.shape).toEqual([2]);
            test_util.expectArraysClose(res, [6, 1]);
        });
        it('sums across axis=0,1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [0, 1]);
            expect(res.shape).toEqual([]);
            test_util.expectArraysClose(res, [7]);
        });
        it('2D, axis=[-1,-2] in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [-1, -2]);
            expect(res.shape).toEqual([]);
            test_util.expectArraysClose(res, [7]);
        });
        it('gradients: sum(2d)', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var dy = ndarray_1.Scalar.new(10);
            var gradients = math.vjp(function () { return math.sum(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [10, 10, 10, 10, 10, 10], 1e-1);
        });
        it('gradients: sum(2d, axis=0)', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [[1, 2], [3, 0], [0, 1]]);
            var dy = ndarray_1.Array1D.new([10, 20]);
            var axis = 0;
            var gradients = math.vjp(function () { return math.sum(a, axis); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [10, 20, 10, 20, 10, 20], 1e-1);
        });
        it('gradients: sum(2d, axis=1)', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [[1, 2], [3, 0], [0, 1]]);
            var dy = ndarray_1.Array1D.new([10, 20, 30]);
            var axis = 1;
            var gradients = math.vjp(function () { return math.sum(a, axis); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [10, 10, 20, 20, 30, 30], 1e-1);
        });
    };
    test_util.describeMathCPU('sum', [tests]);
    test_util.describeMathGPU('sum', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var r = math.mean(a);
            expect(r.dtype).toBe('float32');
            test_util.expectNumbersClose(r.get(), 7 / 6);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
            var r = math.mean(a);
            expect(r.dtype).toBe('float32');
            expect(r.get()).toEqual(NaN);
        });
        it('mean(int32) => float32', function (math) {
            var a = ndarray_1.Array1D.new([1, 5, 7, 3], 'int32');
            var r = math.mean(a);
            expect(r.dtype).toBe('float32');
            test_util.expectNumbersClose(r.get(), 4);
        });
        it('mean(bool) => float32', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true, true], 'bool');
            var r = math.mean(a);
            expect(r.dtype).toBe('float32');
            test_util.expectNumbersClose(r.get(), 3 / 5);
        });
        it('2D array with keep dim', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, null, true);
            expect(res.shape).toEqual([1, 1]);
            expect(res.dtype).toBe('float32');
            test_util.expectArraysClose(res, [7 / 6]);
        });
        it('axis=0 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, [0]);
            expect(res.shape).toEqual([2]);
            expect(res.dtype).toBe('float32');
            test_util.expectArraysClose(res, [4 / 3, 1]);
        });
        it('axis=0 in 2D array, keepDims', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, [0], true);
            expect(res.shape).toEqual([1, 2]);
            expect(res.dtype).toBe('float32');
            test_util.expectArraysClose(res, [4 / 3, 1]);
        });
        it('axis=1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, [1]);
            expect(res.dtype).toBe('float32');
            expect(res.shape).toEqual([3]);
            test_util.expectArraysClose(res, [1.5, 1.5, 0.5]);
        });
        it('axis = -1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, [-1]);
            expect(res.dtype).toBe('float32');
            expect(res.shape).toEqual([3]);
            test_util.expectArraysClose(res, [1.5, 1.5, 0.5]);
        });
        it('2D, axis=1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, 1);
            expect(res.shape).toEqual([2]);
            expect(res.dtype).toBe('float32');
            test_util.expectArraysClose(res, [2, 1 / 3]);
        });
        it('axis=0,1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.mean(a, [0, 1]);
            expect(res.shape).toEqual([]);
            expect(res.dtype).toBe('float32');
            test_util.expectArraysClose(res, [7 / 6]);
        });
        it('gradients', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var dy = ndarray_1.Scalar.new(1.5);
            var vjp = math.vjp(function () { return math.mean(a); }, a, dy);
            expect(vjp.shape).toEqual(a.shape);
            test_util.expectArraysClose(vjp, [
                dy.get() / a.size, dy.get() / a.size, dy.get() / a.size,
                dy.get() / a.size, dy.get() / a.size, dy.get() / a.size
            ]);
        });
        it('gradients throws for defined axis', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var dy = ndarray_1.Scalar.new(1.5);
            expect(function () { return math.vjp(function () { return math.mean(a, 1); }, a, dy); }).toThrowError();
        });
    };
    test_util.describeMathCPU('mean', [tests]);
    test_util.describeMathGPU('mean', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a), mean = _a.mean, variance = _a.variance;
            expect(mean.dtype).toBe('float32');
            expect(variance.dtype).toBe('float32');
            test_util.expectNumbersClose(mean.get(), 7 / 6);
            test_util.expectNumbersClose(variance.get(), 1.1389);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
            var _a = math.moments(a), mean = _a.mean, variance = _a.variance;
            expect(mean.dtype).toBe('float32');
            expect(variance.dtype).toBe('float32');
            expect(mean.get()).toEqual(NaN);
            expect(variance.get()).toEqual(NaN);
        });
        it('moments(int32) => float32', function (math) {
            var a = ndarray_1.Array1D.new([1, 5, 7, 3], 'int32');
            var _a = math.moments(a), mean = _a.mean, variance = _a.variance;
            expect(mean.dtype).toBe('float32');
            expect(variance.dtype).toBe('float32');
            test_util.expectNumbersClose(mean.get(), 4);
            test_util.expectNumbersClose(variance.get(), 5);
        });
        it('moments(bool) => float32', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true, true], 'bool');
            var _a = math.moments(a), mean = _a.mean, variance = _a.variance;
            expect(mean.dtype).toBe('float32');
            expect(variance.dtype).toBe('float32');
            test_util.expectNumbersClose(mean.get(), 3 / 5);
            test_util.expectNumbersClose(variance.get(), 0.23999998);
        });
        it('2D array with keep dim', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a, null, true), mean = _a.mean, variance = _a.variance;
            expect(mean.shape).toEqual([1, 1]);
            expect(mean.dtype).toBe('float32');
            expect(variance.shape).toEqual([1, 1]);
            expect(variance.dtype).toBe('float32');
            test_util.expectArraysClose(mean, [7 / 6]);
            test_util.expectArraysClose(variance, [1.138889]);
        });
        it('axis=0 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a, [0]), mean = _a.mean, variance = _a.variance;
            expect(mean.shape).toEqual([2]);
            expect(mean.dtype).toBe('float32');
            expect(variance.shape).toEqual([2]);
            expect(variance.dtype).toBe('float32');
            test_util.expectArraysClose(mean, [4 / 3, 1]);
            test_util.expectArraysClose(variance, [1.556, 2 / 3]);
        });
        it('axis=1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a, [1]), mean = _a.mean, variance = _a.variance;
            expect(mean.dtype).toBe('float32');
            expect(mean.shape).toEqual([3]);
            expect(variance.dtype).toBe('float32');
            expect(variance.shape).toEqual([3]);
            test_util.expectArraysClose(mean, [1.5, 1.5, 0.5]);
            test_util.expectArraysClose(variance, [0.25, 2.25, 0.25]);
        });
        it('2D, axis=1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a, 1), mean = _a.mean, variance = _a.variance;
            expect(mean.shape).toEqual([2]);
            expect(mean.dtype).toBe('float32');
            expect(variance.shape).toEqual([2]);
            expect(variance.dtype).toBe('float32');
            test_util.expectArraysClose(mean, [2, 1 / 3]);
            test_util.expectArraysClose(variance, [2 / 3, 0.222]);
        });
        it('2D, axis=-1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a, -1), mean = _a.mean, variance = _a.variance;
            expect(mean.shape).toEqual([2]);
            expect(mean.dtype).toBe('float32');
            expect(variance.shape).toEqual([2]);
            expect(variance.dtype).toBe('float32');
            test_util.expectArraysClose(mean, [2, 1 / 3]);
            test_util.expectArraysClose(variance, [2 / 3, 0.222]);
        });
        it('axis=0,1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var _a = math.moments(a, [0, 1]), mean = _a.mean, variance = _a.variance;
            expect(mean.shape).toEqual([]);
            expect(mean.dtype).toBe('float32');
            expect(variance.shape).toEqual([]);
            expect(variance.dtype).toBe('float32');
            test_util.expectArraysClose(mean, [7 / 6]);
            test_util.expectArraysClose(variance, [1.1389]);
        });
    };
    test_util.describeMathCPU('moments', [tests]);
    test_util.describeMathGPU('moments', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('scalar norm', function (math) {
            var a = ndarray_1.Scalar.new(-22.0);
            var norm = math.norm(a);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 22);
        });
        it('vector inf norm', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 3, -4]);
            var norm = math.norm(a, Infinity);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 4);
        });
        it('vector -inf norm', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 3, -4]);
            var norm = math.norm(a, -Infinity);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 1);
        });
        it('vector 1 norm', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 3, -4]);
            var norm = math.norm(a, 1);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 10);
        });
        it('vector euclidean norm', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 3, -4]);
            var norm = math.norm(a, 'euclidean');
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 5.4772);
        });
        it('vector 2-norm', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 3, -4]);
            var norm = math.norm(a, 2);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 5.4772);
        });
        it('vector >2-norm to throw error', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 3, -4]);
            expect(function () { return math.norm(a, 3); }).toThrowError();
        });
        it('matrix inf norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, -3, 1, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 4);
        });
        it('matrix -inf norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, -3, 1, 0, 1]);
            var norm = math.norm(a, -Infinity, [0, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 1);
        });
        it('matrix 1 norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
            var norm = math.norm(a, 1, [0, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 5);
        });
        it('matrix euclidean norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
            var norm = math.norm(a, 'euclidean', [0, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 4.123);
        });
        it('matrix fro norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
            var norm = math.norm(a, 'fro', [0, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectNumbersClose(norm.get(), 4.123);
        });
        it('matrix other norm to throw error', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, -3, 1, 1, 1]);
            expect(function () { return math.norm(a, 2, [0, 1]); }).toThrowError();
        });
        it('propagates NaNs for norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1]);
            expect(norm.dtype).toBe('float32');
            expect(norm.get()).toEqual(NaN);
        });
        it('axis=null in 2D array norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity);
            expect(norm.shape).toEqual([]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('2D array norm with keep dim', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, null, true);
            expect(norm.shape).toEqual([1, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('axis=0 in 2D array norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [0]);
            expect(norm.shape).toEqual([2]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3, 2]);
        });
        it('axis=1 in 2D array norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [1]);
            expect(norm.dtype).toBe('float32');
            expect(norm.shape).toEqual([3]);
            test_util.expectArraysClose(norm, [2, 3, 1]);
        });
        it('axis=1 keepDims in 2D array norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [1], true);
            expect(norm.dtype).toBe('float32');
            expect(norm.shape).toEqual([3, 1]);
            test_util.expectArraysClose(norm, [2, 3, 1]);
        });
        it('2D norm with axis=1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, 1);
            expect(norm.shape).toEqual([2]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3, 1]);
        });
        it('axis=0,1 in 2D array norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1]);
            expect(norm.shape).toEqual([]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('axis=0,1 keepDims in 2D array norm', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1], true);
            expect(norm.shape).toEqual([1, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('3D norm axis=0,1, matrix inf norm', function (math) {
            var a = ndarray_1.Array3D.new([3, 2, 1], [1, 2, -3, 1, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1]);
            expect(norm.shape).toEqual([1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [4]);
        });
        it('axis=0,1 keepDims in 3D array norm', function (math) {
            var a = ndarray_1.Array3D.new([3, 2, 1], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1], true);
            expect(norm.shape).toEqual([1, 1, 1]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('axis=0,1 keepDims in 3D array norm', function (math) {
            var a = ndarray_1.Array3D.new([3, 2, 2], [1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity, [0, 1], true);
            expect(norm.shape).toEqual([1, 1, 2]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [4, 3]);
        });
        it('axis=null in 3D array norm', function (math) {
            var a = ndarray_1.Array3D.new([3, 2, 1], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity);
            expect(norm.shape).toEqual([]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('axis=null in 4D array norm', function (math) {
            var a = ndarray_1.Array4D.new([3, 2, 1, 1], [1, 2, 3, 0, 0, 1]);
            var norm = math.norm(a, Infinity);
            expect(norm.shape).toEqual([]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [3]);
        });
        it('axis=0,1 in 4D array norm', function (math) {
            var a = ndarray_1.Array4D.new([3, 2, 2, 2], [
                1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
                1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
            ]);
            var norm = math.norm(a, Infinity, [0, 1]);
            expect(norm.shape).toEqual([2, 2]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [4, 3, 4, 3]);
        });
        it('axis=0,1 in 4D array norm', function (math) {
            var a = ndarray_1.Array4D.new([3, 2, 2, 2], [
                1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
                1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
            ]);
            var norm = math.norm(a, Infinity, [0, 1], true);
            expect(norm.shape).toEqual([1, 1, 2, 2]);
            expect(norm.dtype).toBe('float32');
            test_util.expectArraysClose(norm, [4, 3, 4, 3]);
        });
    };
    test_util.describeMathCPU('norm', [tests]);
    test_util.describeMathGPU('norm', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=reduction_ops_test.js.map