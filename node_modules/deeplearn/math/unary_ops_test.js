"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1]);
            var result = math.relu(a);
            test_util.expectArraysClose(result, [1, 0, 0, 3, 0]);
        });
        it('does nothing to positive values', function (math) {
            var a = ndarray_1.Scalar.new(1);
            var result = math.relu(a);
            test_util.expectNumbersClose(result.get(), 1);
        });
        it('sets negative values to 0', function (math) {
            var a = ndarray_1.Scalar.new(-1);
            var result = math.relu(a);
            test_util.expectNumbersClose(result.get(), 0);
        });
        it('preserves zero values', function (math) {
            var a = ndarray_1.Scalar.new(0);
            var result = math.relu(a);
            test_util.expectNumbersClose(result.get(), 0);
        });
        it('propagates NaNs, float32', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1, NaN]);
            var result = math.relu(a);
            expect(result.dtype).toBe('float32');
            test_util.expectArraysClose(result, [1, 0, 0, 3, 0, NaN]);
        });
        it('propagates NaNs, int32', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -1, util.NAN_INT32], 'int32');
            var result = math.relu(a);
            expect(result.dtype).toBe('int32');
            test_util.expectArraysClose(result, [1, 0, 0, 3, 0, util.NAN_INT32]);
        });
        it('propagates NaNs, bool', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 0, 1, 0, util.NAN_BOOL], 'bool');
            var result = math.relu(a);
            expect(result.dtype).toBe('bool');
            test_util.expectArraysClose(result, [1, 0, 0, 1, 0, util.NAN_BOOL]);
        });
        it('gradients: positive scalar', function (math) {
            var a = ndarray_1.Scalar.new(3);
            var dy = ndarray_1.Scalar.new(5);
            var gradients = math.vjp(function () { return math.relu(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [5]);
        });
        it('gradients: negative scalar', function (math) {
            var a = ndarray_1.Scalar.new(-3);
            var dy = ndarray_1.Scalar.new(5);
            var gradients = math.vjp(function () { return math.relu(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [0]);
        });
        it('gradients: array', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, -1, -.001, .1]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.relu(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1, 0, 0, 4]);
        });
    };
    test_util.describeMathCPU('relu', [tests]);
    test_util.describeMathGPU('relu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1]);
            var result = math.abs(a);
            test_util.expectArraysClose(result, [1, 2, 0, 3, 0.1]);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1, NaN]);
            var result = math.abs(a);
            test_util.expectArraysClose(result, [1, 2, 0, 3, 0.1, NaN]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(4);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.abs(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 * 1], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3, 5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.abs(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 * 1, 2 * 1, 3 * -1, 4 * 1], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [3, -1, -2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.abs(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 * 1, 2 * -1, 3 * -1, 4 * 1], 1e-1);
        });
    };
    test_util.describeMathCPU('abs', [tests]);
    test_util.describeMathGPU('abs', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('with 1d ndarray', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, -.01, 3, -0.1]);
            var result = math.step(a);
            test_util.expectArraysClose(result, [1, 0, 0, 1, 0]);
        });
        it('with 2d ndarray', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, -5, -3, 4]);
            var result = math.step(a);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result, [1, 0, 0, 1]);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, -.01, 3, NaN]);
            var result = math.step(a);
            test_util.expectArraysClose(result, [1, 0, 0, 1, NaN]);
        });
    };
    test_util.describeMathCPU('step', [tests]);
    test_util.describeMathGPU('step', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, -3, 2, 7, -4]);
            var result = math.neg(a);
            test_util.expectArraysClose(result, [-1, 3, -2, -7, 4]);
        });
        it('propagate NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, -3, 2, 7, NaN]);
            var result = math.neg(a);
            var expected = [-1, 3, -2, -7, NaN];
            test_util.expectArraysClose(result, expected);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(4);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.neg(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 * -1], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3, 5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.neg(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 * -1, 2 * -1, 3 * -1, 4 * -1], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [3, -1, -2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.neg(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 * -1, 2 * -1, 3 * -1, 4 * -1], 1e-1);
        });
    };
    test_util.describeMathCPU('neg', [tests]);
    test_util.describeMathGPU('neg', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.sigmoid(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = 1 / (1 + Math.exp(-values[i]));
            }
            test_util.expectArraysClose(result, expected);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([3, NaN]);
            var res = math.sigmoid(a);
            test_util.expectArraysClose(res, [1 / (1 + Math.exp(-3)), NaN]);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3, 5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.sigmoid(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                var y = 1 / (1 + Math.exp(-a.get(i)));
                expected[i] = dy.get(i) * y * (1 - y);
            }
            test_util.expectArraysClose(gradients, expected, 1e-2);
        });
    };
    test_util.describeMathCPU('sigmoid', [tests]);
    test_util.describeMathGPU('sigmoid', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('sqrt', function (math) {
            var a = ndarray_1.Array1D.new([2, 4]);
            var r = math.sqrt(a);
            test_util.expectNumbersClose(r.get(0), Math.sqrt(2));
            test_util.expectNumbersClose(r.get(1), Math.sqrt(4));
        });
        it('sqrt propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var r = math.sqrt(a);
            test_util.expectArraysClose(r, [Math.sqrt(1), NaN]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(4);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.sqrt(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 / (2 * Math.sqrt(4))], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.sqrt(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                1 / (2 * Math.sqrt(1)), 2 / (2 * Math.sqrt(2)),
                3 / (2 * Math.sqrt(3)), 4 / (2 * Math.sqrt(5))
            ], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [3, 1, 2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.sqrt(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                1 / (2 * Math.sqrt(3)), 2 / (2 * Math.sqrt(1)),
                3 / (2 * Math.sqrt(2)), 4 / (2 * Math.sqrt(3))
            ], 1e-1);
        });
    };
    test_util.describeMathCPU('sqrt', [tests]);
    test_util.describeMathGPU('sqrt', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('1D array', function (math) {
            var a = ndarray_1.Array1D.new([2, 4, Math.sqrt(2)]);
            var r = math.square(a);
            test_util.expectArraysClose(r, [4, 16, 2]);
        });
        it('2D array', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, Math.sqrt(2), Math.sqrt(3)]);
            var r = math.square(a);
            expect(r.shape).toEqual([2, 2]);
            test_util.expectArraysClose(r, [1, 4, 2, 3]);
        });
        it('square propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1.5, NaN]);
            var r = math.square(a);
            test_util.expectArraysClose(r, [2.25, NaN]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.square(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [2 * 5 * 8], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([-1, 2, 3, -5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.square(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [-2, 4 * 2, 6 * 3, -10 * 4], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [-3, 1, 2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.square(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [-6 * 1, 2 * 2, 4 * 3, 6 * 4], 1e-1);
        });
    };
    test_util.describeMathCPU('square', [tests]);
    test_util.describeMathGPU('square', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('log', function (math) {
            var a = ndarray_1.Array1D.new([1, 2]);
            var r = math.log(a);
            test_util.expectNumbersClose(r.get(0), Math.log(1));
            test_util.expectNumbersClose(r.get(1), Math.log(2));
        });
        it('log propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var r = math.log(a);
            test_util.expectArraysClose(r, [Math.log(1), NaN]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var dy = ndarray_1.Scalar.new(3);
            var gradients = math.vjp(function () { return math.log(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [3 / 5], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([-1, 2, 3, -5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.log(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 / -1, 2 / 2, 3 / 3, 4 / -5], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [-3, 1, 2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.log(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 / -3, 2 / 1, 3 / 2, 4 / 3], 1e-1);
        });
    };
    test_util.describeMathCPU('log', [tests]);
    test_util.describeMathGPU('log', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1.5, 2.1, -1.4]);
            var r = math.ceil(a);
            test_util.expectNumbersClose(r.get(0), 2);
            test_util.expectNumbersClose(r.get(1), 3);
            test_util.expectNumbersClose(r.get(2), -1);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1.5, NaN, -1.4]);
            var r = math.ceil(a);
            test_util.expectArraysClose(r, [2, NaN, -1]);
        });
    };
    test_util.describeMathCPU('ceil', [tests]);
    test_util.describeMathGPU('ceil', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1.5, 2.1, -1.4]);
            var r = math.floor(a);
            test_util.expectNumbersClose(r.get(0), 1);
            test_util.expectNumbersClose(r.get(1), 2);
            test_util.expectNumbersClose(r.get(2), -2);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1.5, NaN, -1.4]);
            var r = math.floor(a);
            test_util.expectArraysClose(r, [1, NaN, -2]);
        });
    };
    test_util.describeMathCPU('floor', [tests]);
    test_util.describeMathGPU('floor', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('exp', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 0]);
            var r = math.exp(a);
            test_util.expectNumbersClose(r.get(0), Math.exp(1));
            test_util.expectNumbersClose(r.get(1), Math.exp(2));
            test_util.expectNumbersClose(r.get(2), 1);
        });
        it('exp propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0]);
            var r = math.exp(a);
            test_util.expectArraysClose(r, [Math.exp(1), NaN, 1]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(0.5);
            var dy = ndarray_1.Scalar.new(3);
            var gradients = math.vjp(function () { return math.exp(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [3 * Math.exp(0.5)], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([-1, 2, 3, -5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.exp(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                1 * Math.exp(-1), 2 * Math.exp(2), 3 * Math.exp(3), 4 * Math.exp(-5)
            ], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [-3, 1, 2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.exp(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 * Math.exp(-3), 2 * Math.exp(1), 3 * Math.exp(2), 4 * Math.exp(3)], 1e-1);
        });
    };
    test_util.describeMathCPU('exp', [tests]);
    test_util.describeMathGPU('exp', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.sin(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.sin(values[i]);
            }
            test_util.expectArraysClose(result, expected);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.sin(a);
            test_util.expectArraysClose(res, [Math.sin(4), NaN, Math.sin(0)]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.sin(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 * Math.cos(5)], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([-1, 2, 3, -5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.sin(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                1 * Math.cos(-1), 2 * Math.cos(2), 3 * Math.cos(3), 4 * Math.cos(-5)
            ], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [-3, 1, 2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.sin(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [1 * Math.cos(-3), 2 * Math.cos(1), 3 * Math.cos(2), 4 * Math.cos(3)], 1e-1);
        });
    };
    test_util.describeMathCPU('sin', [tests]);
    test_util.describeMathGPU('sin', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.cos(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.cos(values[i]);
            }
            test_util.expectArraysClose(result, expected);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.cos(a);
            test_util.expectArraysClose(res, [Math.cos(4), NaN, Math.cos(0)]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.cos(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 * Math.sin(5) * -1], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var a = ndarray_1.Array1D.new([-1, 2, 3, -5]);
            var dy = ndarray_1.Array1D.new([1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.cos(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                1 * Math.sin(-1) * -1, 2 * Math.sin(2) * -1, 3 * Math.sin(3) * -1,
                4 * Math.sin(-5) * -1
            ], 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [-3, 1, 2, 3]);
            var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var gradients = math.vjp(function () { return math.cos(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [
                1 * Math.sin(-3) * -1, 2 * Math.sin(1) * -1, 3 * Math.sin(2) * -1,
                4 * Math.sin(3) * -1
            ], 1e-1);
        });
    };
    test_util.describeMathCPU('cos', [tests]);
    test_util.describeMathGPU('cos', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.tan(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.tan(values[i]);
            }
            test_util.expectArraysClose(result, expected, 1e-1);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.tan(a);
            test_util.expectArraysClose(res, [Math.tan(4), NaN, Math.tan(0)]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(0.5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.tan(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 / (Math.cos(0.5) * Math.cos(0.5))], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var aValues = [-1, 2, 3, -5];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array1D.new(aValues);
            var dy = ndarray_1.Array1D.new(dyValues);
            var gradients = math.vjp(function () { return math.tan(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] =
                    dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var aValues = [-3, 1, 2, 3];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array2D.new([2, 2], aValues);
            var dy = ndarray_1.Array2D.new([2, 2], dyValues);
            var gradients = math.vjp(function () { return math.tan(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] =
                    dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
    };
    test_util.describeMathCPU('tan', [tests]);
    test_util.describeMathGPU('tan', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [.1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.asin(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.asin(values[i]);
            }
            test_util.expectArraysClose(result, expected);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.asin(a);
            test_util.expectArraysClose(res, [Math.asin(4), NaN, Math.asin(0)]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(0.5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.asin(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 / Math.sqrt(1 - (0.5 * 0.5))], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var aValues = [-0.1, 0.2, 0.3, -0.5];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array1D.new(aValues);
            var dy = ndarray_1.Array1D.new(dyValues);
            var gradients = math.vjp(function () { return math.asin(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var aValues = [-0.3, 0.1, 0.2, 0.3];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array2D.new([2, 2], aValues);
            var dy = ndarray_1.Array2D.new([2, 2], dyValues);
            var gradients = math.vjp(function () { return math.asin(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
    };
    test_util.describeMathCPU('asin', [tests]);
    test_util.describeMathGPU('asin', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [.1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.acos(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.acos(values[i]);
            }
            test_util.expectArraysClose(result, expected, 1e-1);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.acos(a);
            test_util.expectArraysClose(res, [Math.acos(4), NaN, Math.acos(0)]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(0.5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.acos(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [(-1 * 8) / Math.sqrt(1 - (0.5 * 0.5))], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var aValues = [-0.1, 0.2, 0.3, -0.5];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array1D.new(aValues);
            var dy = ndarray_1.Array1D.new(dyValues);
            var gradients = math.vjp(function () { return math.acos(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] =
                    (-1 * dyValues[i]) / Math.sqrt(1 - (aValues[i] * aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var aValues = [-0.3, 0.1, 0.2, 0.3];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array2D.new([2, 2], aValues);
            var dy = ndarray_1.Array2D.new([2, 2], dyValues);
            var gradients = math.vjp(function () { return math.acos(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] =
                    (-1 * dyValues[i]) / Math.sqrt(1 - (aValues[i] * aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
    };
    test_util.describeMathCPU('acos', [tests]);
    test_util.describeMathGPU('acos', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.atan(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.atan(values[i]);
            }
            test_util.expectArraysClose(result, expected, 1e-3);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.atan(a);
            test_util.expectArraysClose(res, [Math.atan(4), NaN, Math.atan(0)]);
        });
        it('gradients: Scalar', function (math) {
            var a = ndarray_1.Scalar.new(0.5);
            var dy = ndarray_1.Scalar.new(8);
            var gradients = math.vjp(function () { return math.atan(a); }, a, dy);
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [8 / (1 + (0.5 * 0.5))], 1e-1);
        });
        it('gradients: Array1D', function (math) {
            var aValues = [-0.1, 0.2, 0.3, -0.5];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array1D.new(aValues);
            var dy = ndarray_1.Array1D.new(dyValues);
            var gradients = math.vjp(function () { return math.atan(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = dyValues[i] / (1 + (aValues[i] * aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
        it('gradients: Array2D', function (math) {
            var aValues = [-0.3, 0.1, 0.2, 0.3];
            var dyValues = [1, 2, 3, 4];
            var a = ndarray_1.Array2D.new([2, 2], aValues);
            var dy = ndarray_1.Array2D.new([2, 2], dyValues);
            var gradients = math.vjp(function () { return math.atan(a); }, a, dy);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = dyValues[i] / (1 + (aValues[i] * aValues[i]));
            }
            expect(gradients.shape).toEqual(a.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, expected, 1e-1);
        });
    };
    test_util.describeMathCPU('atan', [tests]);
    test_util.describeMathGPU('atan', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var epsilon_1 = 1e-1;
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.sinh(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.sinh(values[i]);
            }
            test_util.expectArraysClose(result, expected, epsilon_1);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.sinh(a);
            test_util.expectArraysClose(res, [Math.sinh(4), NaN, Math.sinh(0)], epsilon_1);
        });
    };
    test_util.describeMathCPU('sinh', [tests]);
    test_util.describeMathGPU('sinh', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var epsilon_2 = 1e-1;
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, -1, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.cosh(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.cosh(values[i]);
            }
            test_util.expectArraysClose(result, expected, epsilon_2);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.cosh(a);
            test_util.expectArraysClose(res, [Math.cosh(4), NaN, Math.cosh(0)], epsilon_2);
        });
    };
    test_util.describeMathCPU('cosh', [tests]);
    test_util.describeMathGPU('cosh', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.tanh(a);
            var expected = [];
            for (var i = 0; i < a.size; i++) {
                expected[i] = util.tanh(values[i]);
            }
            test_util.expectArraysClose(result, expected);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.tanh(a);
            test_util.expectArraysClose(res, [util.tanh(4), NaN, util.tanh(0)]);
        });
    };
    test_util.describeMathCPU('tanh', [tests]);
    test_util.describeMathGPU('tanh', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([0, 1, -2]);
            var result = math.leakyRelu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0, 1, -0.4]);
        });
        it('propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([0, 1, NaN]);
            var result = math.leakyRelu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [0, 1, NaN]);
        });
    };
    test_util.describeMathCPU('leakyRelu', [tests]);
    test_util.describeMathGPU('leakyRelu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('calculate elu', function (math) {
            var a = ndarray_1.Array1D.new([1, -1, 0]);
            var result = math.elu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [1, -0.6321, 0]);
        });
        it('elu propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var result = math.elu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [1, NaN]);
        });
        it('derivative', function (math) {
            var x = ndarray_1.Array1D.new([1, 3, -2]);
            var dy = ndarray_1.Array1D.new([5, 50, 500]);
            var gradients = math.vjp(function () { return math.elu(x); }, x, dy);
            expect(gradients.shape).toEqual(x.shape);
            expect(gradients.dtype).toEqual('float32');
            test_util.expectArraysClose(gradients, [5, 50, 500 * Math.exp(-2)], 1e-1);
        });
    };
    test_util.describeMathCPU('elu', [tests]);
    test_util.describeMathGPU('elu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('calculate selu', function (math) {
            var a = ndarray_1.Array1D.new([1, -1, 0]);
            var result = math.selu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [1.0507, -1.1113, 0]);
        });
        it('selu propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var result = math.selu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result, [1.0507, NaN]);
        });
    };
    test_util.describeMathCPU('selu', [tests]);
    test_util.describeMathGPU('selu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=unary_ops_test.js.map