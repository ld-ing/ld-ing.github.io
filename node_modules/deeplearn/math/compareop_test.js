"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var boolNaN_1 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 1]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 1]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.equal(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.equal(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, boolNaN_1]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, boolNaN_1]);
        });
        it('scalar and 1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 2]);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([6]);
            test_util.expectArraysEqual(res, [0, 1, 0, 0, 0, 1]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0, 0, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 1, 1, 0, 0, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);
        });
        it('broadcasting Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 1, 0, 1, 0, 0]);
        });
        it('broadcasting Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 1, 0, 1, 0, 0]);
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [1, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, boolNaN_1, 1, boolNaN_1, boolNaN_1]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [1.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, 1, boolNaN_1]);
        });
        it('2D and 2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [5, 1]);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysEqual(res, [0, 0, 1, 1, 0, 0]);
        });
        it('2D and scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysEqual(res, [0, 1, 0, 1, 0, 0]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0, 0, 1]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1, 1, 1]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0, 0, 1]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1, 1, 1]);
        });
        it('broadcasting Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        });
        it('broadcasting Array3D shapes - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, 1, 0, 1, boolNaN_1]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, 1, 0, 1, boolNaN_1]);
        });
        it('3D and scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3, 1]);
            test_util.expectArraysEqual(res, [0, 0, 0, 0, 0, 1]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, 0, 0, 0]);
        });
        it('broadcasting Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [1, 0, 0, 0, 1, 0, 0, 0]);
        });
        it('broadcasting Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [1, 0, 0, 0, 1, 0, 0, 0]);
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, 1, boolNaN_1]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            test_util.expectArraysClose(math.equal(a, b), [0, boolNaN_1, 1, boolNaN_1]);
        });
    };
    test_util.describeMathCPU('equal', [tests]);
    test_util.describeMathGPU('equal', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_2 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 1]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 1]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, boolNaN_2]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, boolNaN_2]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0, 0, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 1, 1, 0, 0, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);
        });
        it('mismatch Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatch Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [1, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, boolNaN_2, 1, boolNaN_2, boolNaN_2]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [1.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, 1, boolNaN_2]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0, 0, 1]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1, 1, 1]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0, 0, 1]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1, 1, 1]);
        });
        it('mismatch Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatch Array3D shapes - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, 1, 0, 1, boolNaN_2]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, 1, 0, 1, boolNaN_2]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, 0, 0, 0]);
        });
        it('mismatch Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatch Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            var f = function () {
                math.equalStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, 1, boolNaN_2]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            test_util.expectArraysClose(math.equalStrict(a, b), [0, boolNaN_2, 1, boolNaN_2]);
        });
    };
    test_util.describeMathCPU('equalStrict', [tests]);
    test_util.describeMathGPU('equalStrict', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_3 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 0]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 0]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.notEqual(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.notEqual(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, boolNaN_3]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, boolNaN_3]);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, NaN]);
            var b = ndarray_1.Array1D.new([4, 5, -1]);
            var res = math.notEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysEqual(res, [1, 0, util.NAN_BOOL]);
        });
        it('scalar and 1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 2]);
            var res = math.notEqual(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([6]);
            test_util.expectArraysEqual(res, [1, 0, 1, 1, 1, 0]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 1, 1, 1]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0, 0]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 0, 0, 1, 1, 1]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0, 0]);
        });
        it('broadcasting Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 0, 1, 0, 1, 1]);
        });
        it('broadcasting Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 0, 1, 0, 1, 1]);
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [1, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, boolNaN_3, 0, boolNaN_3, boolNaN_3]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [1.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, 0, boolNaN_3]);
        });
        it('2D and scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.notEqual(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysEqual(res, [1, 0, 1, 0, 1, 1]);
        });
        it('2D and 2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [5, 1]);
            var res = math.notEqual(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysEqual(res, [1, 1, 0, 0, 1, 1]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 1, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0, 0, 0, 0]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 1, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0, 0, 0, 0]);
        });
        it('broadcasting Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
        });
        it('broadcasting Array3D shapes - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, 0, 1, 0, boolNaN_3]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, 0, 1, 0, boolNaN_3]);
        });
        it('3D and scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.notEqual(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3, 1]);
            test_util.expectArraysEqual(res, [1, 1, 1, 1, 1, 0]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 1]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 1, 1, 1, 0, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [0, 1, 1, 1, 0, 1, 1, 1]);
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, 0, boolNaN_3]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            test_util.expectArraysClose(math.notEqual(a, b), [1, boolNaN_3, 0, boolNaN_3]);
        });
    };
    test_util.describeMathCPU('notEqual', [tests]);
    test_util.describeMathGPU('notEqual', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_4 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 0]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 0]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, boolNaN_4]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, boolNaN_4]);
        });
        it('strict version throws when x and y are different shape', function (math) {
            var a = ndarray_1.Array1D.new([2]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.notEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.notEqualStrict(b, a); }).toThrowError();
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 1, 1, 1]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0, 0]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 0, 0, 1, 1, 1]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0, 0]);
        });
        it('mismatch Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatch Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [1, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, boolNaN_4, 0, boolNaN_4, boolNaN_4]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [1.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, 0, boolNaN_4]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [12]]], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 1, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0, 0, 0, 0]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 1, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0, 0, 0, 0]);
        });
        it('mismatch Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatch Array3D shapes - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, 0, 1, 0, boolNaN_4]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, 0, 1, 0, boolNaN_4]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 1]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, 1, 1, 1]);
        });
        it('mismatch Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatch Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            var f = function () {
                math.notEqualStrict(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, 0, boolNaN_4]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 1.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            test_util.expectArraysClose(math.notEqualStrict(a, b), [1, boolNaN_4, 0, boolNaN_4]);
        });
    };
    test_util.describeMathCPU('notEqualStrict', [tests]);
    test_util.describeMathGPU('notEqualStrict', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_5 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 0]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 0]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.less(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.less(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, boolNaN_5]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_5, boolNaN_5]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
        });
        it('broadcasting Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 1]);
        });
        it('broadcasting Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 1]);
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, boolNaN_5, 1, boolNaN_5, boolNaN_5]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, 1, boolNaN_5]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.0]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 1]);
        });
        it('broadcasting Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
        });
        it('broadcasting Array3D float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, 0, 1, 0, boolNaN_5]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, 0, 1, 0, boolNaN_5]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 1, 1, 0, 1, 0, 0]);
        });
        it('broadcasting Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 1, 1, 0, 1, 0, 0]);
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, 1, boolNaN_5]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            var res = math.less(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_5, 1, boolNaN_5]);
        });
    };
    test_util.describeMathCPU('less', [tests]);
    test_util.describeMathGPU('less', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array1D.new([2]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.lessStrict(a, b); }).toThrowError();
            expect(function () { return math.lessStrict(b, a); }).toThrowError();
        });
        it('Array2D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            expect(function () { return math.lessStrict(a, b); }).toThrowError();
            expect(function () { return math.lessStrict(b, a); }).toThrowError();
        });
        it('Array3D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            expect(function () { return math.lessStrict(a, b); }).toThrowError();
            expect(function () { return math.lessStrict(b, a); }).toThrowError();
        });
        it('Array4D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            expect(function () { return math.lessStrict(a, b); }).toThrowError();
            expect(function () { return math.lessStrict(b, a); }).toThrowError();
        });
    };
    test_util.describeMathCPU('lessStrict', [tests]);
    test_util.describeMathGPU('lessStrict', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_6 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.lessEqual(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.lessEqual(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, boolNaN_6]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_6, boolNaN_6]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('broadcasting Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 1, 1, 1, 1]);
        });
        it('broadcasting Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 1, 1, 1, 1]);
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, boolNaN_6, 1, boolNaN_6, boolNaN_6]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, 1, boolNaN_6]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 0]);
        });
        it('broadcasting Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
        });
        it('broadcasting Array3D float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, 1, 1, 1, boolNaN_6]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, 1, 1, 1, boolNaN_6]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1, 0, 0]);
        });
        it('broadcasting Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1, 0, 0]);
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, 1, boolNaN_6]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            var res = math.lessEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_6, 1, boolNaN_6]);
        });
    };
    test_util.describeMathCPU('lessEqual', [tests]);
    test_util.describeMathGPU('lessEqual', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array1D.new([2]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.lessEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.lessEqualStrict(b, a); }).toThrowError();
        });
        it('Array2D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            expect(function () { return math.lessEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.lessEqualStrict(b, a); }).toThrowError();
        });
        it('Array3D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            expect(function () { return math.lessEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.lessEqualStrict(b, a); }).toThrowError();
        });
        it('Array4D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            expect(function () { return math.lessEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.lessEqualStrict(b, a); }).toThrowError();
        });
    };
    test_util.describeMathCPU('lessEqualStrict', [tests]);
    test_util.describeMathGPU('lessEqualStrict', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_7 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0]);
            a = ndarray_1.Array1D.new([3, 3], 'int32');
            b = ndarray_1.Array1D.new([0, 0], 'int32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0]);
            a = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            b = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.greater(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.greater(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, boolNaN_7]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_7, boolNaN_7]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 11]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 11.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
        });
        it('broadcasting Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 0, 0, 0, 0]);
        });
        it('broadcasting Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 0, 0, 0, 0]);
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, boolNaN_7, 0, boolNaN_7, boolNaN_7]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, 0, boolNaN_7]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [11.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 1]);
        });
        it('broadcasting Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
        });
        it('broadcasting Array3D float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, 0, 0, 0, boolNaN_7]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, 0, 0, 0, boolNaN_7]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 8], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 8.1], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 1, 1]);
        });
        it('broadcasting Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 1, 1]);
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, 0, boolNaN_7]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            var res = math.greater(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_7, 0, boolNaN_7]);
        });
    };
    test_util.describeMathCPU('greater', [tests]);
    test_util.describeMathGPU('greater', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array1D.new([2]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.greaterStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterStrict(b, a); }).toThrowError();
        });
        it('Array2D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            expect(function () { return math.greaterStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterStrict(b, a); }).toThrowError();
        });
        it('Array3D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            expect(function () { return math.greaterStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterStrict(b, a); }).toThrowError();
        });
        it('Array4D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            expect(function () { return math.greaterStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterStrict(b, a); }).toThrowError();
        });
    };
    test_util.describeMathCPU('greaterStrict', [tests]);
    test_util.describeMathGPU('greaterStrict', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_8 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 4, 5], 'int32');
            var b = ndarray_1.Array1D.new([2, 3, 5], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 1]);
            a = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            b = ndarray_1.Array1D.new([2, 2, 2], 'int32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1]);
            a = ndarray_1.Array1D.new([0, 0], 'int32');
            b = ndarray_1.Array1D.new([3, 3], 'int32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0]);
        });
        it('Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 4.1, 5.1], 'float32');
            var b = ndarray_1.Array1D.new([2.2, 3.2, 5.1], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 1]);
            a = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            b = ndarray_1.Array1D.new([2.31, 2.31, 2.31], 'float32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1]);
            a = ndarray_1.Array1D.new([0.45, 0.123], 'float32');
            b = ndarray_1.Array1D.new([3.123, 3.321], 'float32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0]);
        });
        it('mismatched Array1D shapes - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 2], 'int32');
            var b = ndarray_1.Array1D.new([1, 2, 3], 'int32');
            var f = function () {
                math.greaterEqual(a, b);
            };
            expect(f).toThrowError();
        });
        it('mismatched Array1D shapes - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([1.1, 2.1, 3.1], 'float32');
            var f = function () {
                math.greaterEqual(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D - int32', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'int32');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, boolNaN_8]);
        });
        it('NaNs in Array1D - float32', function (math) {
            var a = ndarray_1.Array1D.new([1.1, NaN, 2.1], 'float32');
            var b = ndarray_1.Array1D.new([2.1, 3.1, NaN], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, boolNaN_8, boolNaN_8]);
        });
        it('Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 4, 5], [8, 9, 12]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 6], [7, 10, 11]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);
            a = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            b = ndarray_1.Array2D.new([2, 2], [[0, 0], [1, 1]], 'int32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[2.1, 3.1, 6.1], [7.1, 10.1, 11.1]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);
            a = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            b = ndarray_1.Array2D.new([2, 2], [[0.2, 0.2], [1.2, 1.2]], 'float32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
        });
        it('broadcasting Array2D shapes - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[3], [7]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[2, 3, 4], [7, 8, 9]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 0, 1, 0, 0]);
        });
        it('broadcasting Array2D shapes - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 0, 1, 0, 0]);
        });
        it('NaNs in Array2D - int32', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, NaN, 2], [0, NaN, NaN]], 'int32');
            var b = ndarray_1.Array2D.new([2, 3], [[0, NaN, NaN], [1, NaN, 3]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, boolNaN_8, 0, boolNaN_8, boolNaN_8]);
        });
        it('NaNs in Array2D - float32', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1.1, NaN], [0.1, NaN]], 'float32');
            var b = ndarray_1.Array2D.new([2, 2], [[0.1, NaN], [1.1, NaN]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, 0, boolNaN_8]);
        });
        it('Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [4], [5]], [[8], [9], [12]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2], [3], [6]], [[7], [10], [11]]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'int32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
        });
        it('Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [11.1]]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1, 0, 1]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.2]]], 'float32');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], 'float32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1, 1, 1]);
        });
        it('broadcasting Array3D shapes - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[7], [10], [9]]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]);
        });
        it('broadcasting Array3D float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]);
        });
        it('NaNs in Array3D - int32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'int32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, 1, 0, 1, boolNaN_8]);
        });
        it('NaNs in Array3D - float32', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, 1, 0, 1, boolNaN_8]);
        });
        it('Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 4, 5, 8], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 3, 6, 7], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 2, 3], 'int32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'int32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [2, 2, 2, 2], 'int32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
        });
        it('Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 4.1, 5.1, 8.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [2.1, 3.1, 6.1, 7.1], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 1, 0, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 2.2, 3.3], 'float32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 1, 1, 1]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 0.1, 0.1, 0.1], 'float32');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 1.1, 1.1, 1.1], 'float32');
            res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [0, 0, 0, 0]);
        });
        it('broadcasting Array4D shapes - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 5, 9], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 0, 0, 1, 0, 1, 1]);
        });
        it('broadcasting Array4D shapes - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, 0, 0, 0, 1, 0, 1, 1]);
        });
        it('NaNs in Array4D - int32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 0, 0], 'int32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, NaN], 'int32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, 0, boolNaN_8]);
        });
        it('NaNs in Array4D - float32', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, NaN, 0.1, 0.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0.1, 1.1, 1.1, NaN], 'float32');
            var res = math.greaterEqual(a, b);
            expect(res.dtype).toBe('bool');
            test_util.expectArraysClose(res, [1, boolNaN_8, 0, boolNaN_8]);
        });
    };
    test_util.describeMathCPU('greaterEqual', [tests]);
    test_util.describeMathGPU('greaterEqual', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array1D.new([2]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.greaterEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterEqualStrict(b, a); }).toThrowError();
        });
        it('Array2D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1.1], [7.1]], 'float32');
            var b = ndarray_1.Array2D.new([2, 3], [[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], 'float32');
            expect(function () { return math.greaterEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterEqualStrict(b, a); }).toThrowError();
        });
        it('Array3D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [
                [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
                [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
            ], 'float32');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], 'float32');
            expect(function () { return math.greaterEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterEqualStrict(b, a); }).toThrowError();
        });
        it('Array4D - strict version throws when a and b are different shape', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.1, 5.1, 9.1], 'float32');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]], 'float32');
            expect(function () { return math.greaterEqualStrict(a, b); }).toThrowError();
            expect(function () { return math.greaterEqualStrict(b, a); }).toThrowError();
        });
    };
    test_util.describeMathCPU('greaterEqualStrict', [tests]);
    test_util.describeMathGPU('greaterEqualStrict', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=compareop_test.js.map