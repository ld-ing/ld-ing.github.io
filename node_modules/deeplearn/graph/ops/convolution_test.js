"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var dl = require("../../index");
var conv_util = require("../../math/conv_util");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var convolution_1 = require("./convolution");
function assertNoNaNs(t) {
    var values = t.dataSync();
    for (var i = 0; i < values.length; ++i) {
        expect(isNaN(values[i])).toBe(false);
    }
}
describe('Convolution', function () {
    var math = environment_1.ENV.math;
    var wTensor;
    var xTensor;
    var bTensor;
    var yTensor;
    var activations;
    var gradients;
    beforeEach(function () {
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(wTensor);
        activations.disposeArray(xTensor);
        activations.disposeArray(bTensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(wTensor);
        gradients.disposeArray(xTensor);
        gradients.disposeArray(bTensor);
        gradients.disposeArray(yTensor);
    });
    it('Forward prop comparison with convnetjs', function () {
        var inputDepth = 3;
        var outputDepth = 2;
        var fieldSize = 3;
        var stride = 2;
        var zeroPad = 1;
        var weights2D = ndarray_1.Array2D.new([fieldSize * fieldSize * inputDepth, outputDepth], [
            1, -1, 1, 0, -1, 1, -1, 0, -1, 0, 0, 1, -1, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 0, 1, -1, -1, 1, 0, 1, -1, 1, 1, 1, 1, 1, -1,
            -1, 0, 1, 0, 0, 0, 1, -1, -1, -1, 1, 0, -1, 1, 0, -1, 0, 1
        ]);
        var weights = weights2D.as4D(fieldSize, fieldSize, inputDepth, outputDepth);
        var biases = ndarray_1.Array1D.new([1, 0]);
        var x2D = ndarray_1.Array2D.new([25, inputDepth], [
            1, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 0,
            2, 2, 1, 1, 0, 0, 2, 1, 1, 0, 1, 2, 2, 0, 0, 2, 2, 1, 2,
            2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 1,
            0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 1, 0, 2, 0, 0, 0
        ]);
        var x = x2D.as3D(5, 5, inputDepth);
        wTensor = new graph_1.Tensor(weights.shape);
        xTensor = new graph_1.Tensor(x.shape);
        bTensor = new graph_1.Tensor(biases.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fieldSize, outputDepth, stride, zeroPad));
        activations.set(wTensor, weights);
        activations.set(xTensor, x);
        activations.set(bTensor, biases);
        var conv = new convolution_1.Convolution2D(wTensor, xTensor, bTensor, yTensor, fieldSize, outputDepth, stride, zeroPad);
        conv.feedForward(math, activations);
        var result = activations.get(yTensor);
        expect(result.dataSync()).toEqual(new Float32Array([
            7, -8, 8, -2, 7, -2, 5, 5, 4, 6, 1, 2, -1, 3, 7, -2, 1, 4
        ]));
    });
    it('Maintains the rows and cols of input', function () {
        var inputDepth = 3;
        var outputDepth = 2;
        var fSize = 3;
        var stride = 1;
        var weights = dl.randNormal([fSize, fSize, inputDepth, outputDepth]);
        var biases = dl.randNormal([outputDepth]);
        var x = dl.randNormal([5, 5, inputDepth]);
        wTensor = new graph_1.Tensor(weights.shape);
        xTensor = new graph_1.Tensor(x.shape);
        bTensor = new graph_1.Tensor(biases.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride));
        activations.set(wTensor, weights);
        activations.set(xTensor, x);
        activations.set(bTensor, biases);
        var conv = new convolution_1.Convolution2D(wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride);
        conv.feedForward(math, activations);
        var result = activations.get(yTensor);
        expect(result.shape).toEqual([5, 5, outputDepth]);
    });
    it('Can not maintain the rows and cols of input', function () {
        var inputDepth = 3;
        var outputDepth = 2;
        var fSize = 2;
        var stride = 1;
        var weights = dl.randNormal([fSize, fSize, inputDepth, outputDepth]);
        var biases = dl.randNormal([outputDepth]);
        var x = dl.randNormal([5, 5, inputDepth]);
        wTensor = new graph_1.Tensor(weights.shape);
        xTensor = new graph_1.Tensor(x.shape);
        bTensor = new graph_1.Tensor(biases.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride));
        activations.set(wTensor, weights);
        activations.set(xTensor, x);
        activations.set(bTensor, biases);
        var conv = new convolution_1.Convolution2D(wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride);
        conv.feedForward(math, activations);
        var result = activations.get(yTensor);
        expect(result.shape).toEqual([4, 4, outputDepth]);
    });
    it('Large convolution', function () {
        var inputDepth = 3;
        var fSize = 7;
        var outputDepth = 10;
        var stride = 1;
        var zeroPad = 1;
        var weights = dl.randNormal([fSize, fSize, inputDepth, outputDepth]);
        var biases = dl.randNormal([outputDepth]);
        var x = dl.randNormal([30, 30, inputDepth]);
        wTensor = new graph_1.Tensor(weights.shape);
        xTensor = new graph_1.Tensor(x.shape);
        bTensor = new graph_1.Tensor(biases.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride, zeroPad));
        activations.set(wTensor, weights);
        activations.set(xTensor, x);
        activations.set(bTensor, biases);
        var conv = new convolution_1.Convolution2D(wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride, zeroPad);
        conv.feedForward(math, activations);
        var result = activations.get(yTensor);
        assertNoNaNs(result);
        expect(result.shape).toEqual([26, 26, outputDepth]);
    });
    it('simple conv backprop with d1=d2=1 (input and output)', function () {
        var inputDepth = 1;
        var fSize = 2;
        var outputDepth = 1;
        var stride = 1;
        var zeroPad = 0;
        var x3d = dl.randNormal([3, 3, inputDepth]);
        var x = x3d.as2D(3, 3);
        var weights = dl.randNormal([fSize, fSize, inputDepth, outputDepth]);
        var biases = dl.randNormal([outputDepth]);
        wTensor = new graph_1.Tensor(weights.shape);
        xTensor = new graph_1.Tensor(x3d.shape);
        bTensor = new graph_1.Tensor(biases.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x3d.shape, fSize, outputDepth, stride, zeroPad));
        activations.set(wTensor, weights);
        activations.set(xTensor, x3d);
        activations.set(bTensor, biases);
        var conv = new convolution_1.Convolution2D(wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride, zeroPad);
        conv.feedForward(math, activations);
        var y = activations.get(yTensor);
        assertNoNaNs(y);
        test_util.expectNumbersClose(y.get(0, 0, 0), x.get(0, 0) * weights.get(0, 0, 0, 0) +
            x.get(0, 1) * weights.get(0, 1, 0, 0) +
            x.get(1, 0) * weights.get(1, 0, 0, 0) +
            x.get(1, 1) * weights.get(1, 1, 0, 0) + biases.get(0));
        test_util.expectNumbersClose(y.get(0, 1, 0), x.get(0, 1) * weights.get(0, 0, 0, 0) +
            x.get(0, 2) * weights.get(0, 1, 0, 0) +
            x.get(1, 1) * weights.get(1, 0, 0, 0) +
            x.get(1, 2) * weights.get(1, 1, 0, 0) + biases.get(0));
        test_util.expectNumbersClose(y.get(1, 0, 0), x.get(1, 0) * weights.get(0, 0, 0, 0) +
            x.get(1, 1) * weights.get(0, 1, 0, 0) +
            x.get(2, 0) * weights.get(1, 0, 0, 0) +
            x.get(2, 1) * weights.get(1, 1, 0, 0) + biases.get(0));
        test_util.expectNumbersClose(y.get(1, 1, 0), x.get(1, 1) * weights.get(0, 0, 0, 0) +
            x.get(1, 2) * weights.get(0, 1, 0, 0) +
            x.get(2, 1) * weights.get(1, 0, 0, 0) +
            x.get(2, 2) * weights.get(1, 1, 0, 0) + biases.get(0));
        var dy3d = dl.randNormal([2, 2, 1]);
        gradients.add(yTensor, dy3d);
        conv.backProp(math, activations, gradients);
        var dx3d = gradients.get(xTensor);
        var dx = dx3d.as2D(3, 3);
        var dy = dy3d.as2D(2, 2);
        test_util.expectNumbersClose(dx.get(0, 0), dy.get(0, 0) * weights.get(0, 0, 0, 0));
        test_util.expectNumbersClose(dx.get(0, 1), dy.get(0, 0) * weights.get(0, 1, 0, 0) +
            dy.get(0, 1) * weights.get(0, 0, 0, 0));
        test_util.expectNumbersClose(dx.get(0, 2), dy.get(0, 1) * weights.get(0, 1, 0, 0));
        test_util.expectNumbersClose(dx.get(1, 1), dy.get(0, 0) * weights.get(1, 1, 0, 0) +
            dy.get(0, 1) * weights.get(1, 0, 0, 0) +
            dy.get(1, 0) * weights.get(0, 1, 0, 0) +
            dy.get(1, 1) * weights.get(0, 0, 0, 0));
        test_util.expectNumbersClose(dx.get(2, 1), dy.get(1, 0) * weights.get(1, 1, 0, 0) +
            dy.get(1, 1) * weights.get(1, 0, 0, 0));
        var dw = gradients.get(wTensor);
        test_util.expectNumbersClose(dw.get(0, 0, 0, 0), dy.get(0, 0) * x.get(0, 0) + dy.get(0, 1) * x.get(0, 1) +
            dy.get(1, 0) * x.get(1, 0) + dy.get(1, 1) * x.get(1, 1));
        test_util.expectNumbersClose(dw.get(1, 1, 0, 0), dy.get(0, 0) * x.get(1, 1) + dy.get(0, 1) * x.get(1, 2) +
            dy.get(1, 0) * x.get(2, 1) + dy.get(1, 1) * x.get(2, 2));
        var db = gradients.get(bTensor).get(0);
        test_util.expectNumbersClose(db, dy.get(0, 0) + dy.get(0, 1) + dy.get(1, 0) + dy.get(1, 1));
    });
    it('conv backprop with d1=3 d2=7', function () {
        var fSize = 5;
        var inputDepth = 3;
        var outputDepth = 7;
        var stride = 1;
        var zeroPad = 1;
        var weights = dl.randNormal([fSize, fSize, inputDepth, outputDepth]);
        var biases = dl.randNormal([outputDepth]);
        var x = dl.randNormal([10, 10, inputDepth]);
        wTensor = new graph_1.Tensor(weights.shape);
        xTensor = new graph_1.Tensor(x.shape);
        bTensor = new graph_1.Tensor(biases.shape);
        yTensor = new graph_1.Tensor(conv_util.computeOutputShape3D(x.shape, fSize, outputDepth, stride, zeroPad));
        activations.set(wTensor, weights);
        activations.set(xTensor, x);
        activations.set(bTensor, biases);
        var conv = new convolution_1.Convolution2D(wTensor, xTensor, bTensor, yTensor, fSize, outputDepth, stride, zeroPad);
        conv.feedForward(math, activations);
        var result = activations.get(yTensor);
        assertNoNaNs(result);
        var dy = dl.randNormal(result.shape);
        gradients.add(yTensor, dy);
        conv.backProp(math, activations, gradients);
        var dx = gradients.get(xTensor);
        assertNoNaNs(dx);
    });
});
//# sourceMappingURL=convolution_test.js.map