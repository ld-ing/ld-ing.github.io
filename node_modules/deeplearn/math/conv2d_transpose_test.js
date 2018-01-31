"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('input=2x2x1,d2=1,f=2,s=1,p=0', function (math) {
            var origInputDepth = 1;
            var origOutputDepth = 1;
            var inputShape = [1, 1, origOutputDepth];
            var fSize = 2;
            var origPad = 0;
            var origStride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [2]);
            var w = ndarray_1.Array4D.new([fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);
            var result = math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
            var expected = [6, 2, 10, 0];
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, expected);
        });
        it('input=2x2x1,d2=1,f=2,s=1,p=0, batch=2', function (math) {
            var origInputDepth = 1;
            var origOutputDepth = 1;
            var inputShape = [2, 1, 1, origOutputDepth];
            var fSize = 2;
            var origPad = 0;
            var origStride = 1;
            var x = ndarray_1.Array4D.new(inputShape, [2, 3]);
            var w = ndarray_1.Array4D.new([fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);
            var result = math.conv2dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
            var expected = [6, 2, 10, 0, 9, 3, 15, 0];
            expect(result.shape).toEqual([2, 2, 2, 1]);
            test_util.expectArraysClose(result, expected);
        });
        it('throws when x is not rank 3', function (math) {
            var origInputDepth = 1;
            var origOutputDepth = 1;
            var fSize = 2;
            var origPad = 0;
            var origStride = 1;
            var x = ndarray_1.Array2D.new([2, 1], [2, 2]);
            var w = ndarray_1.Array4D.new([fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);
            expect(function () { return math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad); })
                .toThrowError();
        });
        it('throws when weights is not rank 4', function (math) {
            var origInputDepth = 1;
            var origOutputDepth = 1;
            var inputShape = [1, 1, origOutputDepth];
            var fSize = 2;
            var origPad = 0;
            var origStride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [2]);
            var w = ndarray_1.Array3D.new([fSize, fSize, origInputDepth], [3, 1, 5, 0]);
            expect(function () { return math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad); })
                .toThrowError();
        });
        it('throws when x depth does not match weights original output depth', function (math) {
            var origInputDepth = 1;
            var origOutputDepth = 2;
            var wrongOrigOutputDepth = 3;
            var inputShape = [1, 1, origOutputDepth];
            var fSize = 2;
            var origPad = 0;
            var origStride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [2, 2]);
            var w = dl.randNormal([fSize, fSize, origInputDepth, wrongOrigOutputDepth]);
            expect(function () { return math.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad); })
                .toThrowError();
        });
    };
    test_util.describeMathCPU('conv2dTranspose', [tests]);
    test_util.describeMathGPU('conv2dTranspose', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=conv2d_transpose_test.js.map