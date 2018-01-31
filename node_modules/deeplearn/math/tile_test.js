"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('1D (tile)', function (math) {
            var t = ndarray_1.Array1D.new([1, 2, 3]);
            var t2 = math.tile(t, [2]);
            expect(t2.shape).toEqual([6]);
            test_util.expectArraysClose(t2, [1, 2, 3, 1, 2, 3]);
        });
        it('2D (tile)', function (math) {
            var t = ndarray_1.Array2D.new([2, 2], [1, 11, 2, 22]);
            var t2 = math.tile(t, [1, 2]);
            expect(t2.shape).toEqual([2, 4]);
            test_util.expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22]);
            t2 = math.tile(t, [2, 1]);
            expect(t2.shape).toEqual([4, 2]);
            test_util.expectArraysClose(t2, [1, 11, 2, 22, 1, 11, 2, 22]);
            t2 = math.tile(t, [2, 2]);
            expect(t2.shape).toEqual([4, 4]);
            test_util.expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
        });
        it('3D (tile)', function (math) {
            var t = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
            var t2 = math.tile(t, [1, 2, 1]);
            expect(t2.shape).toEqual([2, 4, 2]);
            test_util.expectArraysClose(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
        });
        it('propagates NaNs', function (math) {
            var t = ndarray_1.Array1D.new([1, 2, NaN]);
            var t2 = math.tile(t, [2]);
            expect(t2.shape).toEqual([6]);
            test_util.expectArraysClose(t2, [1, 2, NaN, 1, 2, NaN]);
        });
        it('1D bool (tile)', function (math) {
            var t = ndarray_1.Array1D.new([true, false, true], 'bool');
            var t2 = math.tile(t, [2]);
            expect(t2.shape).toEqual([6]);
            expect(t2.dtype).toBe('bool');
            test_util.expectArraysEqual(t2, [1, 0, 1, 1, 0, 1]);
        });
        it('2D bool (tile)', function (math) {
            var t = ndarray_1.Array2D.new([2, 2], [true, false, true, true], 'bool');
            var t2 = math.tile(t, [1, 2]);
            expect(t2.shape).toEqual([2, 4]);
            expect(t2.dtype).toBe('bool');
            test_util.expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1]);
            t2 = math.tile(t, [2, 1]);
            expect(t2.shape).toEqual([4, 2]);
            expect(t2.dtype).toBe('bool');
            test_util.expectArraysEqual(t2, [1, 0, 1, 1, 1, 0, 1, 1]);
            t2 = math.tile(t, [2, 2]);
            expect(t2.shape).toEqual([4, 4]);
            expect(t2.dtype).toBe('bool');
            test_util.expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
        });
        it('3D bool (tile)', function (math) {
            var t = ndarray_1.Array3D.new([2, 2, 2], [true, false, true, false, true, false, true, false], 'bool');
            var t2 = math.tile(t, [1, 2, 1]);
            expect(t2.shape).toEqual([2, 4, 2]);
            expect(t2.dtype).toBe('bool');
            test_util.expectArraysEqual(t2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        });
        it('bool propagates NaNs', function (math) {
            var t = ndarray_1.Array1D.new([true, false, NaN], 'bool');
            var t2 = math.tile(t, [2]);
            expect(t2.shape).toEqual([6]);
            expect(t2.dtype).toBe('bool');
            test_util.expectArraysEqual(t2, [1, 0, util.getNaN('bool'), 1, 0, util.getNaN('bool')]);
        });
        it('1D int32 (tile)', function (math) {
            var t = ndarray_1.Array1D.new([1, 2, 5], 'int32');
            var t2 = math.tile(t, [2]);
            expect(t2.shape).toEqual([6]);
            expect(t2.dtype).toBe('int32');
            test_util.expectArraysEqual(t2, [1, 2, 5, 1, 2, 5]);
        });
        it('2D int32 (tile)', function (math) {
            var t = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
            var t2 = math.tile(t, [1, 2]);
            expect(t2.shape).toEqual([2, 4]);
            expect(t2.dtype).toBe('int32');
            test_util.expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4]);
            t2 = math.tile(t, [2, 1]);
            expect(t2.shape).toEqual([4, 2]);
            expect(t2.dtype).toBe('int32');
            test_util.expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4]);
            t2 = math.tile(t, [2, 2]);
            expect(t2.shape).toEqual([4, 4]);
            expect(t2.dtype).toBe('int32');
            test_util.expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
        });
        it('3D int32 (tile)', function (math) {
            var t = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8], 'int32');
            var t2 = math.tile(t, [1, 2, 1]);
            expect(t2.shape).toEqual([2, 4, 2]);
            expect(t2.dtype).toBe('int32');
            test_util.expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
        });
        it('int32 propagates NaNs', function (math) {
            var t = ndarray_1.Array1D.new([1, 3, NaN], 'int32');
            var t2 = math.tile(t, [2]);
            expect(t2.shape).toEqual([6]);
            expect(t2.dtype).toBe('int32');
            test_util.expectArraysEqual(t2, [1, 3, util.getNaN('int32'), 1, 3, util.getNaN('int32')]);
        });
    };
    test_util.describeMathCPU('tile', [tests]);
    test_util.describeMathGPU('tile', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=tile_test.js.map