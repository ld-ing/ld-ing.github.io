"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var dl = require("../../index");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var graph_1 = require("../graph");
var session_1 = require("../session");
var adam_optimizer_1 = require("./adam_optimizer");
describe('adam optimizer', function () {
    it('basic', function () {
        var math = environment_1.ENV.math;
        var inputProvider = {
            getNextCopy: function () {
                return ndarray_1.Array1D.new([2, 4]);
            },
            disposeCopy: function (math, example) { }
        };
        math.scope(function () {
            var g = new graph_1.Graph();
            var x = g.placeholder('x', [2]);
            var w = g.variable('w', dl.zeros([1, 2]));
            var b = g.variable('b', dl.zeros([1]));
            var y = g.reduceSum(g.add(g.matmul(w, x), b));
            var optimizer = new adam_optimizer_1.AdamOptimizer(0.1, 0.8, 0.9);
            var session = new session_1.Session(g, math);
            session.train(y, [{ tensor: x, data: inputProvider }], 1, optimizer);
            var dydw = session.activationArrayMap.get(w).dataSync();
            test_util.expectArraysClose(dydw, new Float32Array([-0.1, -0.1]), 1e-5);
            session.train(y, [{ tensor: x, data: inputProvider }], 1, optimizer);
            var dydw2 = session.activationArrayMap.get(w).dataSync();
            test_util.expectArraysClose(dydw2, new Float32Array([-.2, -.2]), 2e-5);
        });
    });
});
//# sourceMappingURL=adam_optimizer_test.js.map