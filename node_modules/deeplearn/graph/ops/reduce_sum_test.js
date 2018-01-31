"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var reduce_sum_1 = require("./reduce_sum");
describe('Reduce sum operation', function () {
    var math = environment_1.ENV.math;
    var reduceSumOp;
    var activations;
    beforeEach(function () {
        activations = new tensor_array_map_1.TensorArrayMap();
    });
    afterEach(function () {
        reduceSumOp.dispose();
        activations.dispose();
    });
    it('Reduces a scalar', function () {
        var xVal = ndarray_1.Scalar.new(-3);
        var x = new graph_1.Tensor(xVal.shape);
        var y = new graph_1.Tensor([]);
        activations.set(x, xVal);
        reduceSumOp = new reduce_sum_1.ReduceSum(x, y);
        reduceSumOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([]);
        expect(yVal.get()).toBe(-3);
    });
    it('Reduces a 1-D tensor', function () {
        var xVal = ndarray_1.Array1D.new([1, 2, 3]);
        var x = new graph_1.Tensor(xVal.shape);
        var y = new graph_1.Tensor([]);
        activations.set(x, xVal);
        reduceSumOp = new reduce_sum_1.ReduceSum(x, y);
        reduceSumOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([]);
        expect(yVal.get()).toBe(6);
    });
    it('Reduces a 2-D tensor', function () {
        var xVal = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var x = new graph_1.Tensor(xVal.shape);
        var y = new graph_1.Tensor([]);
        activations.set(x, xVal);
        reduceSumOp = new reduce_sum_1.ReduceSum(x, y);
        reduceSumOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([]);
        expect(yVal.get()).toBe(21);
    });
});
//# sourceMappingURL=reduce_sum_test.js.map