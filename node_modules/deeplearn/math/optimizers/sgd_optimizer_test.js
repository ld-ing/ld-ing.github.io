"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("../../graph/graph");
var session_1 = require("../../graph/session");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var sgd_optimizer_1 = require("./sgd_optimizer");
var tests = function (it) {
    it('basic', function (math) {
        var learningRate = .1;
        var optimizer = new sgd_optimizer_1.SGDOptimizer(learningRate);
        var x = ndarray_1.variable(ndarray_1.Scalar.new(4));
        var numArrays = math.getNumArrays();
        var cost = optimizer.minimize(function () { return math.square(x); }, true);
        expect(math.getNumArrays()).toBe(numArrays + 1);
        var expectedValue1 = -2 * 4 * learningRate + 4;
        test_util.expectArraysClose(x, [expectedValue1]);
        test_util.expectArraysClose(cost, [Math.pow(4, 2)]);
        cost.dispose();
        numArrays = math.getNumArrays();
        cost = optimizer.minimize(function () { return math.square(x); }, false);
        expect(math.getNumArrays()).toBe(numArrays);
        var expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
        test_util.expectArraysClose(x, [expectedValue2]);
        expect(cost).toBe(null);
        optimizer.dispose();
        x.dispose();
        expect(math.getNumArrays()).toBe(0);
    });
    it('graph', function (math) {
        var inputProvider = {
            getNextCopy: function () {
                return ndarray_1.Array1D.new([2, 4]);
            },
            disposeCopy: function (math, example) { }
        };
        var g = new graph_1.Graph();
        var x = g.placeholder('x', [2]);
        var y = g.square(x);
        var z = g.add(x, g.constant(3));
        var w = g.reduceSum(g.add(y, z));
        var optimizer = new sgd_optimizer_1.SGDOptimizer(0.1);
        var session = new session_1.Session(g, math);
        session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
        var dwdx = session.gradientArrayMap.get(x).dataSync();
        test_util.expectArraysClose(dwdx, new Float32Array([5, 9]), 1e-1);
    });
};
test_util.describeMathCPU('SGDOptimizer', [tests]);
test_util.describeMathGPU('SGDOptimizer', [tests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=sgd_optimizer_test.js.map