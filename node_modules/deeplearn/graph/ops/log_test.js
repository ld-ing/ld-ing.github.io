"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var log_1 = require("./log");
describe('log operation', function () {
    var math = environment_1.ENV.math;
    var xTensor;
    var yTensor;
    var logOp;
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
    it('simple log', function () {
        var x = ndarray_1.Array1D.new([1, 2, 3]);
        xTensor = new graph_1.Tensor(x.shape);
        yTensor = new graph_1.Tensor(x.shape);
        activations.set(xTensor, x);
        logOp = new log_1.Log(xTensor, yTensor);
        logOp.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.shape).toEqual([3]);
        test_util.expectNumbersClose(y.get(0), Math.log(x.get(0)));
        test_util.expectNumbersClose(y.get(1), Math.log(x.get(1)));
        test_util.expectNumbersClose(y.get(2), Math.log(x.get(2)));
        var dy = ndarray_1.Array1D.new([1, 2, 3]);
        gradients.add(yTensor, dy);
        logOp.backProp(math, activations, gradients);
        var dx = gradients.get(xTensor);
        expect(dx.shape).toEqual(dx.shape);
        test_util.expectNumbersClose(dx.get(0), dy.get(0) / x.get(0));
        test_util.expectNumbersClose(dx.get(1), dy.get(1) / x.get(1));
        test_util.expectNumbersClose(dx.get(2), dy.get(2) / x.get(2));
    });
});
//# sourceMappingURL=log_test.js.map