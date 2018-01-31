"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var subtract_1 = require("./subtract");
describe('add operation', function () {
    var math = environment_1.ENV.math;
    var t1;
    var t2;
    var y;
    var subOp;
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
    it('subtracts two 1-D tensors', function () {
        var x1 = ndarray_1.Array1D.new([1, 2, 3]);
        var x2 = ndarray_1.Array1D.new([3, 0, 3]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        activations.set(t1, x1);
        activations.set(t2, x2);
        subOp = new subtract_1.Subtract(t1, t2, y);
        subOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([3]);
        expect(yVal.dataSync()).toEqual(new Float32Array([-2, 2, 0]));
        var dy = ndarray_1.Array1D.new([6, 7, 8]);
        gradients.add(y, dy);
        subOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1);
        var dx2 = gradients.get(t2);
        expect(dx1.shape).toEqual(x1.shape);
        expect(dx1.dataSync()).toEqual(dy.dataSync());
        expect(dx2.shape).toEqual(x2.shape);
        expect(dx2.dataSync()).toEqual(new Float32Array([-6, -7, -8]));
    });
    it('subtracts two 2-D tensors', function () {
        var x1 = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var x2 = ndarray_1.Array2D.new([2, 3], [9, 8, 7, 6, 5, 4]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        activations.set(t1, x1);
        activations.set(t2, x2);
        subOp = new subtract_1.Subtract(t1, t2, y);
        subOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([2, 3]);
        expect(yVal.dataSync()).toEqual(new Float32Array([-8, -6, -4, -2, 0, 2]));
        var dy = ndarray_1.Array2D.new([2, 3], [10, 11, 12, 13, 14, 15]);
        gradients.add(y, dy);
        subOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1);
        var dx2 = gradients.get(t2);
        expect(dx1.shape).toEqual(x1.shape);
        expect(dx1.dataSync()).toEqual(dy.dataSync());
        expect(dx2.shape).toEqual(x2.shape);
        expect(dx2.dataSync()).toEqual(new Float32Array([
            -10, -11, -12, -13, -14, -15
        ]));
    });
    it('ndarray - scalar', function () {
        var x1 = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var x2 = ndarray_1.Scalar.new(2);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        activations.set(t1, x1);
        activations.set(t2, x2);
        subOp = new subtract_1.Subtract(t1, t2, y);
        subOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([2, 3]);
        expect(yVal.dataSync()).toEqual(new Float32Array([-1, 0, 1, 2, 3, 4]));
        var dy = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
        gradients.add(y, dy);
        subOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1);
        var dx2 = gradients.get(t2);
        expect(dx1.shape).toEqual(x1.shape);
        expect(dx1.dataSync()).toEqual(dy.dataSync());
        expect(dx2.shape).toEqual(x2.shape);
        expect(dx2.get()).toEqual(-42);
    });
    it('scalar - ndarray', function () {
        var x1 = ndarray_1.Scalar.new(2);
        var x2 = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        activations.set(t1, x1);
        activations.set(t2, x2);
        subOp = new subtract_1.Subtract(t1, t2, y);
        subOp.feedForward(math, activations);
        var yVal = activations.get(y);
        expect(yVal.shape).toEqual([2, 3]);
        expect(yVal.dataSync()).toEqual(new Float32Array([1, 0, -1, -2, -3, -4]));
        var dy = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
        gradients.add(y, dy);
        subOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(t1);
        var dx2 = gradients.get(t2);
        expect(dx1.shape).toEqual(x1.shape);
        expect(dx1.get()).toEqual(42);
        expect(dx2.shape).toEqual(x2.shape);
        expect(dx2.dataSync()).toEqual(new Float32Array([
            -2, -4, -6, -8, -10, -12
        ]));
    });
    it('throws when shapes of X1 and X2 do not match', function () {
        var x1 = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var x2 = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        activations.set(t1, x1);
        activations.set(t2, x2);
        expect(function () { return new subtract_1.Subtract(t1, t2, y); }).toThrowError();
    });
});
//# sourceMappingURL=subtract_test.js.map