"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('x=[2,2,1] f=[1,1,1,2] s=1 p=0', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 1;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [2]);
            var bias = ndarray_1.Array1D.new([-1]);
            var result = math.conv2d(x, w, bias, stride, pad);
            test_util.expectArraysClose(result, [1, 3, 5, 7]);
        });
        it('x=[2,2,2,1] f=[1,1,1,1] s=1 p=0', function (math) {
            var inputDepth = 1;
            var inShape = [2, 2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 1;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array4D.new(inShape, [1, 2, 3, 4, 5, 6, 7, 8]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [2]);
            var bias = ndarray_1.Array1D.new([-1]);
            var result = math.conv2d(x, w, bias, stride, pad);
            expect(result.shape).toEqual([2, 2, 2, 1]);
            var expected = [1, 3, 5, 7, 9, 11, 13, 15];
            test_util.expectArraysClose(result, expected);
        });
        it('x=[2,2,1] f=[2,2,1,1] s=1 p=0', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            var result = math.conv2d(x, w, bias, stride, pad);
            test_util.expectArraysClose(result, [19]);
        });
        it('throws when x is not rank 3', function (math) {
            var inputDepth = 1;
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
        });
        it('throws when weights is not rank 4', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array3D.new([2, 2, 1], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
        });
        it('throws when biases is not rank 1', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
            var bias = ndarray_1.Array2D.new([2, 2], [2, 2, 2, 2]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
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
            var w = dl.randNormal([fSize, fSize, wrongInputDepth, outputDepth]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
        });
        it('throws when dimRoundingMode is set and pad is not a number', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 'valid';
            var stride = 1;
            var dimRoundingMode = 'round';
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = dl.randNormal([fSize, fSize, inputDepth, outputDepth]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad, dimRoundingMode); })
                .toThrowError();
        });
        it('gradient input=[3,3,1] f=[2,2,1,1] s=1 p=0', function (math) {
            var inputDepth = 1;
            var outputDepth = 1;
            var inputShape = [3, 3, inputDepth];
            var filterSize = 2;
            var stride = 1;
            var pad = 0;
            var filterShape = [filterSize, filterSize, inputDepth, outputDepth];
            var filter = dl.ones(filterShape);
            var bias = ndarray_1.Array1D.new([-1]);
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var dy = ndarray_1.Array3D.new([2, 2, 1], [3, 1, 2, 0]);
            var vjp = math.vjp(function () { return math.conv2d(x, filter, bias, stride, pad); }, { x: x, filter: filter, bias: bias }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [3, 4, 1, 5, 6, 1, 2, 2, 0]);
            expect(vjp.filter.shape).toEqual(filterShape);
            test_util.expectArraysClose(vjp.filter, [13, 19, 31, 37], 1e-1);
            expect(vjp.bias.shape).toEqual(bias.shape);
            test_util.expectArraysClose(vjp.bias, [6], 1e-1);
        });
        it('gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0', function (math) {
            var inputDepth = 1;
            var outputDepth = 1;
            var inputShape = [2, 3, 3, inputDepth];
            var filterSize = 2;
            var stride = 1;
            var pad = 0;
            var filterShape = [filterSize, filterSize, inputDepth, outputDepth];
            var filter = dl.ones(filterShape);
            var bias = ndarray_1.Array1D.new([-1]);
            var x = ndarray_1.Array4D.new(inputShape, [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var dy = ndarray_1.Array4D.new([2, 2, 2, 1], [3, 1, 2, 0, 3, 1, 2, 0]);
            var vjp = math.vjp(function () { return math.conv2d(x, filter, bias, stride, pad); }, { x: x, filter: filter, bias: bias }, dy);
            expect(vjp.x.shape).toEqual(x.shape);
            test_util.expectArraysClose(vjp.x, [3, 4, 1, 5, 6, 1, 2, 2, 0, 3, 4, 1, 5, 6, 1, 2, 2, 0]);
            expect(vjp.filter.shape).toEqual(filterShape);
            test_util.expectArraysClose(vjp.filter, [13 * 2, 19 * 2, 31 * 2, 37 * 2], 1e-1);
            expect(vjp.bias.shape).toEqual(bias.shape);
            test_util.expectArraysClose(vjp.bias, [12]);
        });
    };
    test_util.describeMathCPU('conv2d', [tests]);
    test_util.describeMathGPU('conv2d', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=conv2d_test.js.map