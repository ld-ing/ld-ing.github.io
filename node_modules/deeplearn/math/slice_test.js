"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('slices 1x1 into 1x1 (effectively a copy)', function (math) {
            var a = ndarray_1.Array1D.new([5]);
            var result = math.slice1D(a, 0, 1);
            expect(result.shape).toEqual([1]);
            test_util.expectNumbersClose(result.get(0), 5);
        });
        it('slices 5x1 into shape 2x1 starting at 3', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 4, 5]);
            var result = math.slice1D(a, 3, 2);
            expect(result.shape).toEqual([2]);
            test_util.expectArraysClose(result, [4, 5]);
        });
        it('slices 5x1 into shape 3x1 starting at 1', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 4, 5]);
            var result = math.slice1D(a, 1, 3);
            expect(result.shape).toEqual([3]);
            test_util.expectArraysClose(result, [2, 3, 4]);
        });
    };
    test_util.describeMathCPU('slice1D', [tests]);
    test_util.describeMathGPU('slice1D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('slicing a 1x1 from a 1x1 returns a 1x1', function (math) {
            var a = ndarray_1.Array2D.new([1, 1], [0]);
            var b = math.slice2D(a, [0, 0], [1, 1]);
            expect(b.shape).toEqual([1, 1]);
        });
        it('returns a ndarray of slice size', function (math) {
            var a = dl.zeros([100, 100]);
            var b = math.slice2D(a, [0, 0], [12, 34]);
            expect(b.shape).toEqual([12, 34]);
        });
        it('returns the upper-left submatrix when begin is [0, 0]', function (math) {
            var a = dl.randUniform([10, 10], -1, 1);
            var b = math.slice2D(a, [0, 0], [2, 2]);
            var aValues = a.dataSync();
            test_util.expectArraysClose(b, [aValues[0], aValues[1], aValues[10], aValues[11]]);
        });
        it('returns the rectangle specified', function (math) {
            var a = ndarray_1.Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            var b = math.slice2D(a, [1, 1], [3, 2]);
            test_util.expectArraysClose(b, [5, 6, 8, 9, 11, 12]);
        });
        it('throws when requesting out of bounds slice', function (math) {
            var a = ndarray_1.Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            expect(function () { return math.slice2D(a, [1, 1], [10, 10]); }).toThrowError();
        });
    };
    test_util.describeMathCPU('slice2D', [tests]);
    test_util.describeMathGPU('slice2D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', function (math) {
            var a = ndarray_1.Array3D.new([1, 1, 1], [[[5]]]);
            var result = math.slice3D(a, [0, 0, 0], [1, 1, 1]);
            expect(result.shape).toEqual([1, 1, 1]);
            test_util.expectArraysClose(result, [5]);
        });
        it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
            var result = math.slice3D(a, [1, 0, 0], [1, 2, 2]);
            expect(result.shape).toEqual([1, 2, 2]);
            test_util.expectArraysClose(result, [5, 6, 7, 8]);
        });
        it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
            var result = math.slice3D(a, [0, 1, 1], [2, 1, 1]);
            expect(result.shape).toEqual([2, 1, 1]);
            test_util.expectArraysClose(result, [4, 8]);
        });
    };
    test_util.describeMathCPU('slice3D', [tests]);
    test_util.describeMathGPU('slice3D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', function (math) {
            var a = ndarray_1.Array4D.new([1, 1, 1, 1], [[[[5]]]]);
            var result = math.slice4D(a, [0, 0, 0, 0], [1, 1, 1, 1]);
            expect(result.shape).toEqual([1, 1, 1, 1]);
            test_util.expectArraysClose(result, [5]);
        });
        it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
            var result = math.slice4D(a, [1, 0, 0, 0], [1, 2, 2, 2]);
            expect(result.shape).toEqual([1, 2, 2, 2]);
            test_util.expectArraysClose(result, [11, 22, 33, 44, 55, 66, 77, 88]);
        });
        it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', function (math) {
            var a = ndarray_1.Array4D.new([2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
            var result = math.slice4D(a, [0, 1, 1, 1], [2, 1, 1, 1]);
            expect(result.shape).toEqual([2, 1, 1, 1]);
            test_util.expectArraysClose(result, [8, 88]);
        });
    };
    test_util.describeMathCPU('slice4D', [tests]);
    test_util.describeMathGPU('slice4D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=slice_test.js.map