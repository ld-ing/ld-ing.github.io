"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('1D (gather)', function (math) {
            var t = ndarray_1.Array1D.new([1, 2, 3]);
            var t2 = math.gather(t, ndarray_1.Array1D.new([0, 2, 0, 1], 'int32'), 0);
            expect(t2.shape).toEqual([4]);
            test_util.expectArraysClose(t2, [1, 3, 1, 2]);
        });
        it('2D (gather)', function (math) {
            var t = ndarray_1.Array2D.new([2, 2], [1, 11, 2, 22]);
            var t2 = math.gather(t, ndarray_1.Array1D.new([1, 0, 0, 1], 'int32'), 0);
            expect(t2.shape).toEqual([4, 2]);
            test_util.expectArraysClose(t2, [2, 22, 1, 11, 1, 11, 2, 22]);
            t2 = math.gather(t, ndarray_1.Array1D.new([1, 0, 0, 1], 'int32'), 1);
            expect(t2.shape).toEqual([2, 4]);
            test_util.expectArraysClose(t2, [11, 1, 1, 11, 22, 2, 2, 22]);
        });
        it('3D (gather)', function (math) {
            var t = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
            var t2 = math.gather(t, ndarray_1.Array1D.new([1, 0, 0, 1], 'int32'), 2);
            expect(t2.shape).toEqual([2, 2, 4]);
            test_util.expectArraysClose(t2, [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
        });
        it('bool (gather)', function (math) {
            var t = ndarray_1.Array1D.new([true, false, true], 'bool');
            var t2 = math.gather(t, ndarray_1.Array1D.new([0, 2, 0, 1], 'int32'), 0);
            expect(t2.shape).toEqual([4]);
            expect(t2.dtype).toBe('bool');
            expect(t2.getValues()).toEqual(new Uint8Array([1, 1, 1, 0]));
        });
        it('int32 (gather)', function (math) {
            var t = ndarray_1.Array1D.new([1, 2, 5], 'int32');
            var t2 = math.gather(t, ndarray_1.Array1D.new([0, 2, 0, 1], 'int32'), 0);
            expect(t2.shape).toEqual([4]);
            expect(t2.dtype).toBe('int32');
            expect(t2.getValues()).toEqual(new Int32Array([1, 5, 1, 2]));
        });
        it('propagates NaNs', function (math) {
            var t = ndarray_1.Array1D.new([1, 2, NaN]);
            var t2 = math.gather(t, ndarray_1.Array1D.new([0, 2, 0, 1], 'int32'), 0);
            expect(t2.shape).toEqual([4]);
            test_util.expectArraysClose(t2, [1, NaN, 1, 2]);
        });
    };
    test_util.describeMathCPU('gather', [tests]);
    test_util.describeMathGPU('gather', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=gather_test.js.map