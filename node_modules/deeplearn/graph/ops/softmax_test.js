"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var softmax_1 = require("./softmax");
describe('softmax cross entropy cost', function () {
    var math = environment_1.ENV.math;
    var logitsTensor;
    var labelTensor;
    var yTensor;
    var activations;
    var gradients;
    beforeEach(function () {
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(logitsTensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(logitsTensor);
        gradients.disposeArray(yTensor);
    });
    it('matches theory', function () {
        var logits = ndarray_1.Array1D.new([1, 2, 3]);
        var label = ndarray_1.Array1D.new([0.3, 0.6, 0.1]);
        var softmaxLogits = math.softmax(logits);
        logitsTensor = new graph_1.Tensor(logits.shape);
        labelTensor = new graph_1.Tensor(label.shape);
        yTensor = new graph_1.Tensor([]);
        activations.set(logitsTensor, logits);
        activations.set(labelTensor, label);
        var op = new softmax_1.SoftmaxCrossEntropyCost(logitsTensor, labelTensor, yTensor);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        test_util.expectNumbersClose(y.get(0), -Math.log(softmaxLogits.get(0)) * label.get(0) +
            -Math.log(softmaxLogits.get(1)) * label.get(1) +
            -Math.log(softmaxLogits.get(2)) * label.get(2));
        var dy = ndarray_1.Scalar.new(1);
        gradients.add(yTensor, dy);
        op.backProp(math, activations, gradients);
        var dLogits = gradients.get(logitsTensor);
        test_util.expectNumbersClose(dLogits.get(0), softmaxLogits.get(0) - label.get(0));
        test_util.expectNumbersClose(dLogits.get(1), softmaxLogits.get(1) - label.get(1));
        test_util.expectNumbersClose(dLogits.get(2), softmaxLogits.get(2) - label.get(2));
    });
});
describe('softmax operation', function () {
    var math = environment_1.ENV.math;
    var logitsTensor;
    var yTensor;
    var activations;
    var gradients;
    beforeEach(function () {
        var math = environment_1.ENV.math;
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(logitsTensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(logitsTensor);
        gradients.disposeArray(yTensor);
    });
    it('matches theory', function () {
        var logits = ndarray_1.Array1D.new([10, 0, -1]);
        var softmaxLogits = math.softmax(logits);
        logitsTensor = new graph_1.Tensor(logits.shape);
        yTensor = new graph_1.Tensor([]);
        activations.set(logitsTensor, logits);
        var op = new softmax_1.Softmax(logitsTensor, yTensor);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        test_util.expectArraysClose(y.dataSync(), softmaxLogits.dataSync());
        var dy = ndarray_1.Array1D.new([1, 10, 100]);
        gradients.add(yTensor, dy);
        op.backProp(math, activations, gradients);
        var dLogits = gradients.get(logitsTensor);
        var sumLogitsdLogits = math.sum(math.elementWiseMul(dy, softmaxLogits));
        test_util.expectNumbersClose(dLogits.get(0), (dy.get(0) - sumLogitsdLogits.get(0)) * softmaxLogits.get(0));
        test_util.expectNumbersClose(dLogits.get(1), (dy.get(1) - sumLogitsdLogits.get(0)) * softmaxLogits.get(1));
        test_util.expectNumbersClose(dLogits.get(2), (dy.get(2) - sumLogitsdLogits.get(0)) * softmaxLogits.get(2));
    });
});
//# sourceMappingURL=softmax_test.js.map