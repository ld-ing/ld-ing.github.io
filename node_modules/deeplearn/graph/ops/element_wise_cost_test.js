"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var element_wise_cost_1 = require("./element_wise_cost");
describe('MeanSquaredCost', function () {
    var math = environment_1.ENV.math;
    var x1Tensor;
    var x2Tensor;
    var yTensor;
    var meanSquaredCostOperation;
    var activations;
    var gradients;
    beforeEach(function () {
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(x1Tensor);
        activations.disposeArray(x2Tensor);
        activations.disposeArray(yTensor);
    });
    it('mean squared cost, forward & backward', function () {
        var x1 = ndarray_1.Array1D.new([1, 2, 3, 4]);
        var x2 = ndarray_1.Array1D.new([2, 4, 6, 8]);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor([]);
        activations.set(x1Tensor, x1);
        activations.set(x2Tensor, x2);
        meanSquaredCostOperation = new element_wise_cost_1.MeanSquaredCost(x1Tensor, x2Tensor, yTensor);
        meanSquaredCostOperation.feedForward(math, activations);
        meanSquaredCostOperation.backProp(math, activations, gradients);
        var y = activations.get(yTensor);
        expect(y.shape).toEqual([]);
        expect(y.dataSync()).toEqual(new Float32Array([30 / 8]));
        var dx1 = gradients.get(x1Tensor);
        var dx2 = gradients.get(x2Tensor);
        expect(dx1.shape).toEqual(x1.shape);
        expect(dx2.shape).toEqual(x2.shape);
        expect(dx1.dataSync()).toEqual(new Float32Array([-1, -2, -3, -4]));
        expect(dx2.dataSync()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
});
//# sourceMappingURL=element_wise_cost_test.js.map