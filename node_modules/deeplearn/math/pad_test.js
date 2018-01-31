"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('Should pad 1D arrays', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 6], 'int32');
            var b = math.pad1D(a, [2, 3]);
            test_util.expectArraysClose(b, [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
        });
        it('Should not pad 1D arrays with 0s', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 4], 'int32');
            var b = math.pad1D(a, [0, 0]);
            test_util.expectArraysClose(b, [1, 2, 3, 4]);
        });
        it('Should handle padding with custom value', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 4], 'int32');
            var b = math.pad1D(a, [2, 3], 9);
            test_util.expectArraysClose(b, [9, 9, 1, 2, 3, 4, 9, 9, 9]);
            a = ndarray_1.Array1D.new([1, 2, 3, 4]);
            b = math.pad1D(a, [2, 1], 1.1);
            test_util.expectArraysClose(b, [1.1, 1.1, 1, 2, 3, 4, 1.1]);
            a = ndarray_1.Array1D.new([1, 2, 3, 4]);
            b = math.pad1D(a, [2, 1], 1);
            test_util.expectArraysClose(b, [1, 1, 1, 2, 3, 4, 1]);
        });
        it('Should handle NaNs with 1D arrays', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 2, NaN]);
            var b = math.pad1D(a, [1, 1]);
            test_util.expectArraysClose(b, [0, 1, NaN, 2, NaN, 0]);
        });
        it('Should handle invalid paddings', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3, 4], 'int32');
            var f = function () {
                math.pad1D(a, [2, 2, 2]);
            };
            expect(f).toThrowError();
        });
    };
    test_util.describeMathCPU('pad1D', [tests]);
    test_util.describeMathGPU('pad1D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Should pad 2D arrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1], [2]], 'int32');
            var b = math.pad2D(a, [[1, 1], [1, 1]]);
            test_util.expectArraysClose(b, [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);
            a = ndarray_1.Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]], 'int32');
            b = math.pad2D(a, [[2, 2], [1, 1]]);
            test_util.expectArraysClose(b, [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
                0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]);
        });
        it('Should not pad 2D arrays with 0s', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]], 'int32');
            var b = math.pad2D(a, [[0, 0], [0, 0]]);
            test_util.expectArraysClose(b, [1, 2, 3, 4, 5, 6]);
        });
        it('Should handle padding with custom value', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]], 'int32');
            var b = math.pad2D(a, [[1, 1], [1, 1]], 10);
            test_util.expectArraysClose(b, [
                10, 10, 10, 10, 10, 10, 1, 2, 3, 10,
                10, 4, 5, 6, 10, 10, 10, 10, 10, 10
            ]);
            a = ndarray_1.Array2D.new([2, 1], [[1], [1]]);
            b = math.pad2D(a, [[1, 1], [1, 1]], -2.1);
            test_util.expectArraysClose(b, [-2.1, -2.1, -2.1, -2.1, 1, -2.1, -2.1, 1, -2.1, -2.1, -2.1, -2.1]);
            a = ndarray_1.Array2D.new([2, 1], [[1], [1]]);
            b = math.pad2D(a, [[1, 1], [1, 1]], -2);
            test_util.expectArraysClose(b, [-2, -2, -2, -2, 1, -2, -2, 1, -2, -2, -2, -2]);
        });
        it('Should handle NaNs with 2D arrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [[1, NaN], [1, NaN]]);
            var b = math.pad2D(a, [[1, 1], [1, 1]]);
            test_util.expectArraysClose(b, [0, 0, 0, 0, 0, 1, NaN, 0, 0, 1, NaN, 0, 0, 0, 0, 0]);
        });
        it('Should handle invalid paddings', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [[1], [2]], 'int32');
            var f = function () {
                math.pad2D(a, [[2, 2, 2], [1, 1, 1]]);
            };
            expect(f).toThrowError();
        });
    };
    test_util.describeMathCPU('pad2D', [tests]);
    test_util.describeMathGPU('pad2D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=pad_test.js.map