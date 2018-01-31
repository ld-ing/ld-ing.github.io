"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var argmaxequals_1 = require("./argmaxequals");
describe('Argmax equals oper', function () {
    var math = environment_1.ENV.math;
    var t1;
    var t2;
    var y;
    var argmaxEqualsOp;
    var tensorArrayMap;
    beforeEach(function () {
        tensorArrayMap = new tensor_array_map_1.TensorArrayMap();
    });
    afterEach(function () {
        tensorArrayMap.disposeArray(t1);
        tensorArrayMap.disposeArray(t2);
        tensorArrayMap.disposeArray(y);
    });
    it('argmax equals', function () {
        var x1 = ndarray_1.Array1D.new([0, 2, 1]);
        var x2 = ndarray_1.Array1D.new([2, 4, 3]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        tensorArrayMap.set(t1, x1);
        tensorArrayMap.set(t2, x2);
        argmaxEqualsOp = new argmaxequals_1.ArgMaxEquals(t1, t2, y);
        argmaxEqualsOp.feedForward(math, tensorArrayMap);
        var yVal = tensorArrayMap.get(y);
        expect(yVal.shape).toEqual([]);
        expect(yVal.dataSync()).toEqual(new Uint8Array([1]));
    });
    it('argmax not equals', function () {
        var x1 = ndarray_1.Array1D.new([0, 2, 1]);
        var x2 = ndarray_1.Array1D.new([5, 4, 3]);
        t1 = new graph_1.Tensor(x1.shape);
        t2 = new graph_1.Tensor(x2.shape);
        y = new graph_1.Tensor(x1.shape);
        tensorArrayMap.set(t1, x1);
        tensorArrayMap.set(t2, x2);
        argmaxEqualsOp = new argmaxequals_1.ArgMaxEquals(t1, t2, y);
        argmaxEqualsOp.feedForward(math, tensorArrayMap);
        var yVal = tensorArrayMap.get(y);
        expect(yVal.shape).toEqual([]);
        expect(yVal.dataSync()).toEqual(new Uint8Array([0]));
    });
});
//# sourceMappingURL=argmaxequals_test.js.map