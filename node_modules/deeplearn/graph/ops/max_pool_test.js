"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var dl = require("../../index");
var conv_util = require("../../math/conv_util");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var max_pool_1 = require("./max_pool");
describe('Max pool', function () {
    var math = environment_1.ENV.math;
    var xTensor;
    var yTensor;
    var activations;
    var gradients;
    beforeEach(function () {
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(xTensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(xTensor);
        gradients.disposeArray(yTensor);
    });
    it('Simple MaxPool', function () {
        var fSize = 2;
        var stride = 1;
        var pad = 0;
        var depth = 1;
        var x = ndarray_1.Array3D.new([3, 3, depth], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
        xTensor = new graph_1.Tensor(x.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], stride, pad));
        activations.set(xTensor, x);
        var op = new max_pool_1.MaxPool(xTensor, yTensor, fSize, stride, pad);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        var expectedResult = ndarray_1.Array3D.new([2, 2, depth], [5, 6, 9, 9]);
        test_util.expectArraysClose(y, expectedResult);
        var dy = ndarray_1.Array3D.new([2, 2, depth], [50, 60, 90, 80]);
        gradients.add(yTensor, dy);
        op.backProp(math, activations, gradients);
        var dx = gradients.get(xTensor);
        var expectedBackprop = ndarray_1.Array3D.new([3, 3, depth], [0, 0, 0, 0, 50, 60, 0, 170, 0]);
        test_util.expectArraysClose(dx, expectedBackprop);
    });
    it('MaxPool depth = 2', function () {
        var fSize = 2;
        var stride = 2;
        var pad = 0;
        var depth = 2;
        var x = ndarray_1.Array3D.new([4, 4, depth], [
            1, 11, 2, 22, 3, 33, 4, 44, 5, 55, 6, 66, 7, 77, 8, 88,
            9, 99, 10, 100, 11, 110, 12, 120, 13, 130, 14, 140, 15, 150, 16, 160
        ]);
        xTensor = new graph_1.Tensor(x.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], stride, pad));
        activations.set(xTensor, x);
        var op = new max_pool_1.MaxPool(xTensor, yTensor, fSize, stride, pad);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        var expectedResult = ndarray_1.Array3D.new([2, 2, 2], [6, 66, 8, 88, 14, 140, 16, 160]);
        test_util.expectArraysClose(y.dataSync(), expectedResult.dataSync());
    });
    it('MaxPool depth = 2, with some negative numbers', function () {
        var fSize = 2;
        var stride = 2;
        var pad = 0;
        var x = ndarray_1.Array3D.new([4, 4, 2], [
            -1, 11, 2, 22, 3, 33, 4, 44, 5, 55, 6, -66, 7, -77, 8, 88,
            9, 99, 10, 100, -11, 110, 12, 120, 13, 130, 14, 140, 15, 150, 16, -160
        ]);
        xTensor = new graph_1.Tensor(x.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], stride, pad));
        activations.set(xTensor, x);
        var op = new max_pool_1.MaxPool(xTensor, yTensor, fSize, stride, pad);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        var expectedResult = ndarray_1.Array3D.new([2, 2, 2], [6, 55, 8, 88, 14, 140, 16, 150]);
        test_util.expectArraysClose(y.dataSync(), expectedResult.dataSync());
    });
    it('MaxPool downsampling depth is preserved', function () {
        var fSize = 2;
        var stride = 2;
        var pad = 0;
        var x = dl.randNormal([6, 6, 5]);
        xTensor = new graph_1.Tensor(x.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, x.shape[2], stride, pad));
        activations.set(xTensor, x);
        var op = new max_pool_1.MaxPool(xTensor, yTensor, fSize, stride, pad);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.shape).toEqual([3, 3, 5]);
    });
});
//# sourceMappingURL=max_pool_test.js.map