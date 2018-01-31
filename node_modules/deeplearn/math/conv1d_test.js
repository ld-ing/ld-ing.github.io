"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('conv1d input=2x2x1,d2=1,f=1,s=1,p=same', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 1;
            var pad = 'same';
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array3D.new([fSize, inputDepth, outputDepth], [3]);
            var bias = ndarray_1.Array1D.new([0]);
            var result = math.conv1d(x, w, bias, stride, pad);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result, [3, 6, 9, 12]);
        });
        it('conv1d input=4x1,d2=1,f=2x1x1,s=1,p=valid', function (math) {
            var inputDepth = 1;
            var inputShape = [4, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 'valid';
            var stride = 1;
            var x = ndarray_1.Array2D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array3D.new([fSize, inputDepth, outputDepth], [2, 1]);
            var bias = ndarray_1.Array1D.new([0]);
            var result = math.conv1d(x, w, bias, stride, pad);
            expect(result.shape).toEqual([3, 1]);
            test_util.expectArraysClose(result, [4, 7, 10]);
        });
        it('throws when x is not rank 3', function (math) {
            var inputDepth = 1;
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var w = ndarray_1.Array3D.new([fSize, inputDepth, outputDepth], [3, 1]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv1d(x, w, bias, stride, pad); }).toThrowError();
        });
        it('throws when weights is not rank 3', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([2, 2, 1, 1], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv1d(x, w, bias, stride, pad); }).toThrowError();
        });
        it('throws when biases is not rank 1', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array3D.new([fSize, inputDepth, outputDepth], [3, 1]);
            var bias = ndarray_1.Array2D.new([2, 2], [2, 2, 2, 2]);
            expect(function () { return math.conv1d(x, w, bias, stride, pad); }).toThrowError();
        });
        it('throws when x depth does not match weight depth', function (math) {
            var inputDepth = 1;
            var wrongInputDepth = 5;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = dl.randNormal([fSize, wrongInputDepth, outputDepth]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv1d(x, w, bias, stride, pad); }).toThrowError();
        });
    };
    test_util.describeMathCPU('conv1d', [tests]);
    test_util.describeMathGPU('conv1d', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=conv1d_test.js.map