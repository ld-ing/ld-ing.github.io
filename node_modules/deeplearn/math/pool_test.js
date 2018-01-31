"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('x=[1,1,1] f=[1,1] s=1 [0] => [0]', function (math) {
            var x = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var result = math.maxPool(x, 1, 1, 0);
            test_util.expectArraysClose(result, [0]);
        });
        it('x=[3,3,1] f=[2,2] s=1', function (math) {
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var result = math.maxPool(x, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [5, 6, 9, 9]);
        });
        it('x=[2,3,3,1] f=[2,2] s=1', function (math) {
            var x = ndarray_1.Array4D.new([2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var result = math.maxPool(x, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2, 1]);
            test_util.expectArraysClose(result, [5, 6, 9, 9, 5, 6, 8, 9]);
        });
        it('[x=[3,3,1] f=[2,2] s=1 propagates NaNs', function (math) {
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 9]);
            var result = math.maxPool(x, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [5, 6, NaN, NaN]);
        });
        it('x=[3,3,2] f=[2,2] s=1', function (math) {
            var x = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
            var result = math.maxPool(x, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2]);
            test_util.expectArraysClose(result, [5, 99, 6, 88, 9, 66, 9, 55]);
        });
        it('x=[4,4,1] f=[2,2] s=2', function (math) {
            var x = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var result = math.maxPool(x, 2, 2, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [5, 7, 13, 15]);
        });
        it('x=[2,2,1] f=[2,2] s=2 p=1', function (math) {
            var x = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var result = math.maxPool(x, 2, 2, 1);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [1, 2, 3, 4]);
        });
        it('throws when x is not rank 3', function (math) {
            var x = ndarray_1.Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            expect(function () { return math.maxPool(x, 2, 1, 0); }).toThrowError();
        });
        it('throws when dimRoundingMode is set and pad is not a number', function (math) {
            var x = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var pad = 'valid';
            var dimRoundingMode = 'round';
            expect(function () { return math.maxPool(x, 2, 1, pad, dimRoundingMode); }).toThrowError();
        });
        it('gradients x=[3,3,1] f=[2,2] s=1 no dup max value, test #1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var expected = [0, 0, 0, 0, 1, 2, 0, 3, 4];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradients x=[3,3,1] f=[2,2] s=1 no dup max value, test #2', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [9, 5, 6, 6, 8, 4, 9, 5, 10]);
            var expected = [1, 0, 0, 0, 2, 0, 3, 0, 4];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradients x=[2,3,3,1] f=[2,2] s=1 no duplicate max value', function (math) {
            var dy = ndarray_1.Array4D.new([2, 2, 2, 1], [1, 2, 3, 4, 1, 2, 3, 4]);
            var x = ndarray_1.Array4D.new([2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 5, 6, 6, 8, 4, 9, 5, 10]);
            var expected = [0, 0, 0, 0, 1, 2, 0, 3, 4, 1, 0, 0, 0, 2, 0, 3, 0, 4];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[3,3,1] f=[2,2] s=1 dup max value, test 1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [0, 0, 0, 0, 5, 0, 0, 0, 0]);
            var expected = [0, 0, 0, 0, 10, 0, 0, 0, 0];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[3,3,1] f=[2,2] s=1 dup max value, test 2', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 3, 2, 1, 2, 1, 1, 1, 5]);
            var expected = [0, 3, 0, 0, 3, 0, 0, 0, 4];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[2,3,3,1] f=[2,2] s=1 dup max value in 2nd input', function (math) {
            var dy = ndarray_1.Array4D.new([2, 2, 2, 1], [1, 2, 3, 4, 5, 6, 7, 8]);
            var x = ndarray_1.Array4D.new([2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var expected = new Float32Array([0, 0, 0, 0, 1, 2, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 15, 0]);
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[4,4,1] f=[2,2] s=2 test #1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var expected = [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 2, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[4,4,1] f=[2,2] s=2 test #2', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([4, 4, 1], [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1]);
            var expected = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 2, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[5,5,1] f=[3,3] s=2 no duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([5, 5, 1], [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ]);
            var expected = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4
            ];
            var vjp = math.vjp(function () { return math.maxPool(x, 3, 2, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[5,5,1] f=[3,3] s=2 duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([5, 5, 1], [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12
            ]);
            var expected = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ];
            var vjp = math.vjp(function () { return math.maxPool(x, 3, 2, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[3,3,2] f=[2,2] s=1, no duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
            var x = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 55, 3, 66, 4, 66, 5, 88, 6, 44, 7, 99, 8, 55, 9, 100]);
            var expected = [0, 44, 0, 0, 0, 0, 0, 0, 1, 33, 2, 0, 0, 22, 3, 0, 4, 11];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[3,3,2] f=[2,2] s=1 duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
            var x = ndarray_1.Array3D.new([3, 3, 2], [0, 1, 0, 3, 0, 2, 0, 1, 5, 2, 0, 1, 0, 1, 0, 1, 0, 5]);
            var expected = new Float32Array([0, 0, 0, 77, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 11]);
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=[4,4,2] f=[2,2] s=1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
            var x = ndarray_1.Array3D.new([4, 4, 2], [
                0, 1, 1, 2, 2, 2, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1,
                8, 1, 9, 1, 10, 1, 11, 1, 12, 1, 13, 2, 14, 2, 15, 1
            ]);
            var expected = [
                0, 0, 0, 11, 0, 22, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 33, 0, 44, 4, 0
            ];
            var vjp = math.vjp(function () { return math.maxPool(x, 2, 2, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
        it('gradient x=5x5x2, f=3, s=2 no duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
            var x = ndarray_1.Array3D.new([5, 5, 2], [
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
                8, 9, 9, 10, 10, 11, 11, 12, 24, 13, 13, 14, 14, 15, 15, 16, 16,
                17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 12
            ]);
            var expected = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 110, 0, 0, 2, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0
            ];
            var vjp = math.vjp(function () { return math.maxPool(x, 3, 2, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, expected);
        });
    };
    test_util.describeMathCPU('maxPool', [tests]);
    test_util.describeMathGPU('maxPool', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', function (math) {
            var a = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var result = math.minPool(a, 1, 1, 0);
            test_util.expectArraysClose(result, [0]);
        });
        it('3x3x1 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [1, 2, 4, 5]);
        });
        it('3x3x1 in, 2x2 filter, 1 stride, batch=2', function (math) {
            var a = ndarray_1.Array4D.new([2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 5, 4, 6, 7, 9, 8]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2, 1]);
            test_util.expectArraysClose(result, [1, 2, 4, 5, 1, 2, 4, 4]);
        });
        it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [1, 2, NaN, NaN]);
        });
        it('3x3x2 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2]);
            test_util.expectArraysClose(result, [1, 55, 2, 44, 4, 22, 5, 11]);
        });
        it('4x4x1 in, 2x2 filter, 2 stride', function (math) {
            var a = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var result = math.minPool(a, 2, 2, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [0, 2, 8, 10]);
        });
        it('2x2x1 in, 2x2 filter, 2 stride, pad=1', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var result = math.minPool(a, 2, 2, 1);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [1, 2, 3, 4]);
        });
        it('throws when dimRoundingMode is set and pad is not a number', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var pad = 'valid';
            var dimRoundingMode = 'round';
            expect(function () { return math.minPool(a, 2, 1, pad, dimRoundingMode); }).toThrowError();
        });
    };
    test_util.describeMathCPU('minPool', [tests]);
    test_util.describeMathGPU('minPool', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('x=[1,1,1] f=[1,1] s=1 [0] => [0]', function (math) {
            var a = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var result = math.avgPool(a, 1, 1, 0);
            test_util.expectArraysClose(result, [0]);
        });
        it('x=[3,3,1] f=[2,2] s=1', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            expect(result.dtype).toBe('float32');
            test_util.expectArraysClose(result, [3, 4, 6.25, 7]);
        });
        it('x=[3,3,1] f=[2,2] s=1 input int32, output float32', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8], 'int32');
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            expect(result.dtype).toBe('float32');
            test_util.expectArraysClose(result, [3, 4, 6.25, 7]);
        });
        it('x=[2,3,3,1] f=[2,2], s=1', function (math) {
            var a = ndarray_1.Array4D.new([2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2, 1]);
            test_util.expectArraysClose(result, [3, 4, 6.25, 7, 3, 4, 6, 7]);
        });
        it('x=[3,3,1] f=[2,2] s=1 propagates NaNs', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [3, 4, NaN, NaN]);
        });
        it('x=[3,3,2] f=[2,2] s=1', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2]);
            test_util.expectArraysClose(result, [3, 77, 4, 66, 6.25, 44, 7, 33]);
        });
        it('x=[4,4,1] f=[2,2] s=2', function (math) {
            var a = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var result = math.avgPool(a, 2, 2, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [2.5, 4.5, 10.5, 12.5]);
        });
        it('x=[2,2,1] f=[2,2] s=2 p=1', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var result = math.avgPool(a, 2, 2, 1);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [0.25, 0.5, 0.75, 1]);
        });
        it('gradient x=[1,1,1] f=[1,1] s=1 [0] => [0]', function (math) {
            var x = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var dy = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var vjp = math.vjp(function () {
                return math.avgPool(x, 1, 1, 0);
            }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [0]);
        });
        it('gradient x=[3,3,1] f=[2,2] s=1', function (math) {
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var avgMultiplier = 1 / (2 * 2);
            var vjp = math.vjp(function () { return math.avgPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [
                1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
                4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
                3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier
            ]);
        });
        it('gradient x=[2,3,3,1] f=[2,2], s=1', function (math) {
            var x = ndarray_1.Array4D.new([2, 3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var dy = ndarray_1.Array4D.new([2, 2, 2, 1], [1, 2, 3, 4, 1, 2, 3, 4]);
            var avgMultiplier = 1 / (2 * 2);
            var vjp = math.vjp(function () { return math.avgPool(x, 2, 1, 0); }, { x: x }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [
                1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
                4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
                3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier,
                1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
                4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
                3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier
            ]);
        });
        it('throws when dimRoundingMode is set and pad is not a number', function (math) {
            var x = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var pad = 'valid';
            var dimRoundingMode = 'round';
            expect(function () { return math.avgPool(x, 2, 1, pad, dimRoundingMode); }).toThrowError();
        });
    };
    test_util.describeMathCPU('avgPool', [tests]);
    test_util.describeMathGPU('avgPool', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=pool_test.js.map