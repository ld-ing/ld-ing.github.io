"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var boolNaN_1 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D.', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 0], 'bool');
            var b = ndarray_1.Array1D.new([0, 1, 0], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([0, 0, 0], 'bool');
            b = ndarray_1.Array1D.new([0, 0, 0], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([1, 1], 'bool');
            b = ndarray_1.Array1D.new([1, 1], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1]);
        });
        it('mismatched Array1D shapes', function (math) {
            var a = ndarray_1.Array1D.new([1, 0], 'bool');
            var b = ndarray_1.Array1D.new([0, 1, 0], 'bool');
            var f = function () {
                math.logicalAnd(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'bool');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN_1, boolNaN_1]);
        });
        it('Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
            var b = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0, 0, 0]);
            a = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
            b = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
        });
        it('broadcasting Array2D shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1], [0]], 'bool');
            var b = ndarray_1.Array2D.new([2, 3], [[0, 1, 0], [0, 1, 0]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 1, 0, 0, 0, 0]);
        });
        it('NaNs in Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
            var b = ndarray_1.Array2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN_1, 0, boolNaN_1]);
        });
        it('Array3D', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [1]]], 'bool');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0, 0, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 1, 1, 1]);
        });
        it('broadcasting Array3D shapes', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]], 'bool');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
        });
        it('NaNs in Array3D', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN_1, 1, 0, 0, boolNaN_1]);
        });
        it('Array4D', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 1, 0], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [1, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [1, 0, 0, 0, 0, 0, 0, 0]);
        });
        it('NaNs in Array4D', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
            test_util.expectArraysClose(math.logicalAnd(a, b), [0, boolNaN_1, 0, boolNaN_1]);
        });
    };
    test_util.describeMathCPU('logicalAnd', [tests]);
    test_util.describeMathGPU('logicalAnd', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var boolNaN_2 = util.getNaN('bool');
    var tests = function (it) {
        it('Array1D.', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 0], 'bool');
            var b = ndarray_1.Array1D.new([0, 1, 0], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 0]);
            a = ndarray_1.Array1D.new([0, 0, 0], 'bool');
            b = ndarray_1.Array1D.new([0, 0, 0], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0]);
            a = ndarray_1.Array1D.new([1, 1], 'bool');
            b = ndarray_1.Array1D.new([1, 1], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 1]);
        });
        it('mismatched Array1D shapes', function (math) {
            var a = ndarray_1.Array1D.new([1, 0], 'bool');
            var b = ndarray_1.Array1D.new([0, 1, 0], 'bool');
            var f = function () {
                math.logicalOr(a, b);
            };
            expect(f).toThrowError();
        });
        it('NaNs in Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0], 'bool');
            var b = ndarray_1.Array1D.new([0, 0, NaN], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN_2, boolNaN_2]);
        });
        it('Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 0, 1], [0, 0, 0]], 'bool');
            var b = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 0, 1, 0]);
            a = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
            b = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [1, 1, 1]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
        });
        it('broadcasting Array2D shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1], [0]], 'bool');
            var b = ndarray_1.Array2D.new([2, 3], [[0, 0, 0], [0, 1, 0]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0, 1, 0]);
        });
        it('NaNs in Array2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1, NaN], [0, NaN]], 'bool');
            var b = ndarray_1.Array2D.new([2, 2], [[0, NaN], [1, NaN]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN_2, 1, boolNaN_2]);
        });
        it('Array3D', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 1, 1, 0, 0]);
            a = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
            b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [0]], [[1], [1], [1]]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 1, 1, 1]);
        });
        it('broadcasting Array3D shapes', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 2], [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]], 'bool');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [0]]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]);
        });
        it('NaNs in Array3D', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [NaN], [1]], [[0], [0], [0]]], 'bool');
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[0], [0], [1]], [[1], [0], [NaN]]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN_2, 1, 1, 0, boolNaN_2]);
        });
        it('Array4D', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 0, 0], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 0, 0, 0], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [0, 0, 0, 0]);
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
            b = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 1, 1, 1], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 1, 1]);
        });
        it('broadcasting Array4D shapes', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 0, 1, 0], 'bool');
            var b = ndarray_1.Array4D.new([2, 2, 1, 2], [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, 1, 0, 0, 1, 1, 1, 1]);
        });
        it('NaNs in Array4D', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, NaN, 1, 0], 'bool');
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [0, 1, 0, NaN], 'bool');
            test_util.expectArraysClose(math.logicalOr(a, b), [1, boolNaN_2, 1, boolNaN_2]);
        });
    };
    test_util.describeMathCPU('logicalOr', [tests]);
    test_util.describeMathGPU('logicalOr', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            var a = ndarray_1.Array1D.new([10, 10, 10, 10]);
            var b = ndarray_1.Array1D.new([20, 20, 20, 20]);
            test_util.expectArraysClose(math.where(c, a, b), [10, 20, 10, 20]);
        });
        it('Array1D different a/b shapes', function (math) {
            var c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            var a = ndarray_1.Array1D.new([10, 10, 10]);
            var b = ndarray_1.Array1D.new([20, 20, 20, 20]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            a = ndarray_1.Array1D.new([10, 10, 10, 10]);
            b = ndarray_1.Array1D.new([20, 20, 20]);
            f = function () {
                math.where(c, a, b);
            };
        });
        it('Array1D different condition/a shapes', function (math) {
            var c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            var a = ndarray_1.Array1D.new([10, 10, 10]);
            var b = ndarray_1.Array1D.new([20, 20, 20]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array2D', function (math) {
            var c = ndarray_1.Array2D.new([2, 2], [[1, 0], [0, 1]], 'bool');
            var a = ndarray_1.Array2D.new([2, 2], [[10, 10], [10, 10]]);
            var b = ndarray_1.Array2D.new([2, 2], [[5, 5], [5, 5]]);
            test_util.expectArraysClose(math.where(c, a, b), [10, 5, 5, 10]);
        });
        it('Array2D different a/b shapes', function (math) {
            var c = ndarray_1.Array2D.new([2, 2], [[1, 1], [0, 0]], 'bool');
            var a = ndarray_1.Array2D.new([2, 3], [[5, 5, 5], [5, 5, 5]]);
            var b = ndarray_1.Array2D.new([2, 2], [[4, 4], [4, 4]]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            c = ndarray_1.Array2D.new([2, 2], [[1, 1], [0, 0]], 'bool');
            a = ndarray_1.Array2D.new([2, 2], [[5, 5], [5, 5]]);
            b = ndarray_1.Array2D.new([2, 3], [[4, 4, 4], [4, 4, 4]]);
            f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array2D different condition/a shapes', function (math) {
            var c = ndarray_1.Array2D.new([2, 2], [[1, 0], [0, 1]], 'bool');
            var a = ndarray_1.Array2D.new([2, 3], [[10, 10, 10], [10, 10, 10]]);
            var b = ndarray_1.Array2D.new([2, 3], [[5, 5, 5], [5, 5, 5]]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array2D different `a` dimension w/ condition rank=1', function (math) {
            var c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            var a = ndarray_1.Array2D.new([2, 2], [[10, 10], [10, 10]]);
            var b = ndarray_1.Array2D.new([2, 2], [[5, 5], [5, 5]]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            a = ndarray_1.Array2D.new([4, 1], [[10], [10], [10], [10]]);
            b = ndarray_1.Array2D.new([4, 1], [[5], [5], [5], [5]]);
            test_util.expectArraysClose(math.where(c, a, b), [10, 5, 10, 5]);
            a = ndarray_1.Array2D.new([4, 2], [[10, 10], [10, 10], [10, 10], [10, 10]]);
            b = ndarray_1.Array2D.new([4, 2], [[5, 5], [5, 5], [5, 5], [5, 5]]);
            test_util.expectArraysClose(math.where(c, a, b), [10, 10, 5, 5, 10, 10, 5, 5]);
        });
        it('Array3D', function (math) {
            var c = ndarray_1.Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
            test_util.expectArraysClose(math.where(c, a, b), [5, 3, 5, 3, 3, 3]);
        });
        it('Array3D different a/b shapes', function (math) {
            var c = ndarray_1.Array3D.new([2, 3, 1], [[[1], [0], [1]], [[0], [0], [0]]], 'bool');
            var a = ndarray_1.Array3D.new([2, 2, 1], [[[5], [5]], [[5], [5]]]);
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            a = ndarray_1.Array3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
            b = ndarray_1.Array3D.new([2, 2, 1], [[[3], [3]], [[3], [3]]]);
            f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array3D different condition/a shapes', function (math) {
            var c = ndarray_1.Array3D.new([2, 2, 1], [[[1], [0]], [[0], [0]]], 'bool');
            var a = ndarray_1.Array3D.new([2, 3, 1], [[[5], [5], [5]], [[5], [5], [5]]]);
            var b = ndarray_1.Array3D.new([2, 3, 1], [[[3], [3], [3]], [[3], [3], [3]]]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array3D different `a` dimension w/ condition rank=1', function (math) {
            var c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            var a = ndarray_1.Array3D.new([2, 2, 2], [[[9, 9], [9, 9]], [[9, 9], [9, 9]]]);
            var b = ndarray_1.Array3D.new([2, 2, 2], [[[8, 8], [8, 8]], [[8, 8], [8, 8]]]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            a = ndarray_1.Array3D.new([4, 1, 1], [[[9]], [[9]], [[9]], [[9]]]);
            b = ndarray_1.Array3D.new([4, 1, 1], [[[8]], [[8]], [[8]], [[8]]]);
            test_util.expectArraysClose(math.where(c, a, b), [9, 8, 9, 8]);
            a = ndarray_1.Array3D.new([4, 2, 1], [[[9], [9]], [[9], [9]], [[9], [9]], [[9], [9]]]);
            b = ndarray_1.Array3D.new([4, 2, 1], [[[8], [8]], [[8], [8]], [[8], [8]], [[8], [8]]]);
            test_util.expectArraysClose(math.where(c, a, b), [9, 9, 8, 8, 9, 9, 8, 8]);
        });
        it('Array4D', function (math) {
            var c = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 0, 1, 1], 'bool');
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
            test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 7]);
        });
        it('Array4D different a/b shapes', function (math) {
            var c = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 0, 1, 1], 'bool');
            var a = ndarray_1.Array4D.new([2, 2, 2, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            a = ndarray_1.Array4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
            b = ndarray_1.Array4D.new([2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
            f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array4D different condition/a shapes', function (math) {
            var c = ndarray_1.Array4D.new([2, 2, 2, 1], [1, 0, 1, 1, 1, 0, 1, 1], 'bool');
            var a = ndarray_1.Array4D.new([2, 2, 1, 1], [7, 7, 7, 7]);
            var b = ndarray_1.Array4D.new([2, 2, 1, 1], [3, 3, 3, 3]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
        });
        it('Array4D different `a` dimension w/ condition rank=1', function (math) {
            var c = ndarray_1.Array1D.new([1, 0, 1, 0], 'bool');
            var a = ndarray_1.Array4D.new([2, 2, 2, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
            var b = ndarray_1.Array4D.new([2, 2, 2, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
            var f = function () {
                math.where(c, a, b);
            };
            expect(f).toThrowError();
            a = ndarray_1.Array4D.new([4, 1, 1, 1], [7, 7, 7, 7]);
            b = ndarray_1.Array4D.new([4, 1, 1, 1], [3, 3, 3, 3]);
            test_util.expectArraysClose(math.where(c, a, b), [7, 3, 7, 3]);
            a = ndarray_1.Array4D.new([4, 2, 1, 1], [7, 7, 7, 7, 7, 7, 7, 7]);
            b = ndarray_1.Array4D.new([4, 2, 1, 1], [3, 3, 3, 3, 3, 3, 3, 3]);
            test_util.expectArraysClose(math.where(c, a, b), [7, 7, 3, 3, 7, 7, 3, 3]);
        });
    };
    test_util.describeMathCPU('where', [tests]);
    test_util.describeMathGPU('where', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=logicalop_test.js.map