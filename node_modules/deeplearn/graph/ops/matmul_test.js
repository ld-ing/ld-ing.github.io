"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var matmul_1 = require("./matmul");
describe('add operation', function () {
    var math = environment_1.ENV.math;
    var t1;
    var t2;
    var y;
    var matmulOp;
    var activations;
    var gradients;
    beforeEach(function () {
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(t1);
        activations.disposeArray(t2);
        activations.disposeArray(y);
        gradients.disposeArray(t1);
        gradients.disposeArray(t2);
        gradients.disposeArray(y);
    });
    it('matmul two NDArray2Ds', function () {
        var x1 = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 10, 20, 30]);
        var x2 = ndarray_1.Array2D.new([3, 2], [2, 3, 4, 1, 2, 3]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor([x1.shape[0], x2.shape[1]]);
        activations.set(t1, x1);
        activations.set(t2, x2);
        matmulOp = new matmul_1.MatMul(t1, t2, y);
        matmulOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([x1.shape[0], x2.shape[1]]);
        expect(yVal.get(0, 0))
            .toEqual(x1.get(0, 0) * x2.get(0, 0) + x1.get(0, 1) * x2.get(1, 0) +
            x1.get(0, 2) * x2.get(2, 0));
        expect(yVal.get(0, 1))
            .toEqual(x1.get(0, 0) * x2.get(0, 1) + x1.get(0, 1) * x2.get(1, 1) +
            x1.get(0, 2) * x2.get(2, 1));
        expect(yVal.get(1, 0))
            .toEqual(x1.get(1, 0) * x2.get(0, 0) + x1.get(1, 1) * x2.get(1, 0) +
            x1.get(1, 2) * x2.get(2, 0));
        expect(yVal.get(1, 1))
            .toEqual(x1.get(1, 0) * x2.get(0, 1) + x1.get(1, 1) * x2.get(1, 1) +
            x1.get(1, 2) * x2.get(2, 1));
        var dy = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        gradients.add(y, dy);
        matmulOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1);
        expect(dx1.shape).toEqual(x1.shape);
        expect(dx1.get(0, 0))
            .toEqual(dy.get(0, 0) * x2.get(0, 0) + dy.get(0, 1) * x2.get(0, 1));
        expect(dx1.get(0, 1))
            .toEqual(dy.get(0, 0) * x2.get(1, 0) + dy.get(0, 1) * x2.get(1, 1));
        expect(dx1.get(0, 2))
            .toEqual(dy.get(0, 0) * x2.get(2, 0) + dy.get(0, 1) * x2.get(2, 1));
        expect(dx1.get(1, 0))
            .toEqual(dy.get(1, 0) * x2.get(0, 0) + dy.get(1, 1) * x2.get(0, 1));
        expect(dx1.get(1, 1))
            .toEqual(dy.get(1, 0) * x2.get(1, 0) + dy.get(1, 1) * x2.get(1, 1));
        expect(dx1.get(1, 2))
            .toEqual(dy.get(1, 0) * x2.get(2, 0) + dy.get(1, 1) * x2.get(2, 1));
        var dx2 = gradients.get(t2);
        expect(dx2.shape).toEqual(x2.shape);
        expect(dx2.get(0, 0))
            .toEqual(x1.get(0, 0) * dy.get(0, 0) + x1.get(1, 0) * dy.get(1, 0));
        expect(dx2.get(0, 1))
            .toEqual(x1.get(0, 0) * dy.get(0, 1) + x1.get(1, 0) * dy.get(1, 1));
        expect(dx2.get(1, 0))
            .toEqual(x1.get(0, 1) * dy.get(0, 0) + x1.get(1, 1) * dy.get(1, 0));
        expect(dx2.get(1, 1))
            .toEqual(x1.get(0, 1) * dy.get(0, 1) + x1.get(1, 1) * dy.get(1, 1));
        expect(dx2.get(2, 0))
            .toEqual(x1.get(0, 2) * dy.get(0, 0) + x1.get(1, 2) * dy.get(1, 0));
        expect(dx2.get(2, 1))
            .toEqual(x1.get(0, 2) * dy.get(0, 1) + x1.get(1, 2) * dy.get(1, 1));
    });
    it('matrix times vector', function () {
        var inputSize = 3;
        var outputSize = 2;
        var x1 = ndarray_1.Array2D.new([outputSize, inputSize], [1, 2, 0, 4, 3, 2]);
        var x2 = ndarray_1.Array1D.new([1, 2, 3]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor([x1.shape[0], x2.shape[1]]);
        activations.set(t1, x1);
        activations.set(t2, x2);
        var op = new matmul_1.MatMul(t1, t2, y);
        op.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.get(0)).toBe(5);
        expect(yVal.get(1)).toBe(16);
        var dy = ndarray_1.Array1D.new([2, 3]);
        gradients.add(y, dy);
        op.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1).as2D(x1.shape[0], x1.shape[1]);
        expect(dx1.get(0, 0)).toBe(dy.get(0) * x2.get(0));
        expect(dx1.get(0, 1)).toBe(dy.get(0) * x2.get(1));
        expect(dx1.get(0, 2)).toBe(dy.get(0) * x2.get(2));
        expect(dx1.get(1, 0)).toBe(dy.get(1) * x2.get(0));
        expect(dx1.get(1, 1)).toBe(dy.get(1) * x2.get(1));
        expect(dx1.get(1, 2)).toBe(dy.get(1) * x2.get(2));
        var dx2 = gradients.get(t2).as1D();
        expect(dx2.get(0))
            .toBe(x1.get(0, 0) * dy.get(0) + x1.get(1, 0) * dy.get(1));
        expect(dx2.get(1))
            .toBe(x1.get(0, 1) * dy.get(0) + x1.get(1, 1) * dy.get(1));
        expect(dx2.get(2))
            .toBe(x1.get(0, 2) * dy.get(0) + x1.get(1, 2) * dy.get(1));
    });
    it('vector times matrix', function () {
        var x1 = ndarray_1.Array1D.new([1, 2, 3]);
        var x2 = ndarray_1.Array2D.new([3, 2], [1, 2, 0, 4, 3, 2]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor([x1.shape[0], x2.shape[1]]);
        activations.set(t1, x1);
        activations.set(t2, x2);
        var op = new matmul_1.MatMul(t1, t2, y);
        op.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.get(0)).toBe(10);
        expect(yVal.get(1)).toBe(16);
        var dy = ndarray_1.Array1D.new([2, 3]);
        gradients.add(y, dy);
        op.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1).as1D();
        expect(dx1.get(0))
            .toBe(dy.get(0) * x2.get(0, 0) + dy.get(1) * x2.get(0, 1));
        expect(dx1.get(1))
            .toBe(dy.get(0) * x2.get(1, 0) + dy.get(1) * x2.get(1, 1));
        expect(dx1.get(2))
            .toBe(dy.get(0) * x2.get(2, 0) + dy.get(1) * x2.get(2, 1));
        var dx2 = gradients.get(t2).as2D(x2.shape[0], x2.shape[1]);
        expect(dx2.get(0, 0)).toBe(x1.get(0) * dy.get(0));
        expect(dx2.get(0, 1)).toBe(x1.get(0) * dy.get(1));
        expect(dx2.get(1, 0)).toBe(x1.get(1) * dy.get(0));
        expect(dx2.get(1, 1)).toBe(x1.get(1) * dy.get(1));
        expect(dx2.get(2, 0)).toBe(x1.get(2) * dy.get(0));
        expect(dx2.get(2, 1)).toBe(x1.get(2) * dy.get(1));
    });
});
//# sourceMappingURL=matmul_test.js.map