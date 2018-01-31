"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var ndarray_1 = require("../math/ndarray");
var graph_1 = require("./graph");
var graph_util = require("./graph_util");
var tensor_array_map_1 = require("./tensor_array_map");
var TestNode = (function (_super) {
    __extends(TestNode, _super);
    function TestNode() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    TestNode.prototype.validate = function () { };
    return TestNode;
}(graph_1.Node));
describe('graph_util.getUnorderedEvaluationSet', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('returns only node in graph', function () {
        var n = new TestNode(g, '', {}, new graph_1.Tensor([]));
        var path = graph_util.getUnorderedEvaluationSet([n], []);
        expect(path.length).toEqual(1);
        expect(path[0]).toBe(n);
    });
    it('returns both nodes in graph with two connected nodes', function () {
        var t = new graph_1.Tensor([]);
        var s = new TestNode(g, '', {}, t);
        var e = new TestNode(g, '', { 't': t }, new graph_1.Tensor([]));
        var path = graph_util.getUnorderedEvaluationSet([e], []);
        expect(path.length).toEqual(2);
        expect(path).toContain(s);
        expect(path).toContain(e);
    });
    it('adds nodes in the termination set', function () {
        var t0 = new graph_1.Tensor([]);
        var n0 = new TestNode(g, '', {}, t0);
        var t1 = new graph_1.Tensor([]);
        var n1 = new TestNode(g, '', { 't0': t0 }, t1);
        var n2 = new TestNode(g, '', { 't1': t1 }, new graph_1.Tensor([]));
        var path = graph_util.getUnorderedEvaluationSet([n2], [n0]);
        expect(path.length).toEqual(3);
        expect(path).toContain(n0);
        expect(path).toContain(n1);
        expect(path).toContain(n2);
    });
    it('does not process inputs from nodes in the termination set', function () {
        var t0 = new graph_1.Tensor([]);
        var t1 = new graph_1.Tensor([]);
        var n1 = new TestNode(g, '', { 't0': t0 }, t1);
        var n2 = new TestNode(g, '', { 't1': t1 }, new graph_1.Tensor([]));
        var path = graph_util.getUnorderedEvaluationSet([n2], [n1]);
        expect(path.length).toEqual(2);
        expect(path).toContain(n1);
        expect(path).toContain(n2);
    });
    it('accumulates multiple inputs from nodes', function () {
        var t0 = new graph_1.Tensor([]);
        var i0 = new TestNode(g, '', {}, t0);
        var t1 = new graph_1.Tensor([]);
        var i1 = new TestNode(g, '', {}, t1);
        var n = new TestNode(g, '', { 't0': t0, 't1': t1 }, new graph_1.Tensor([]));
        var path = graph_util.getUnorderedEvaluationSet([n], []);
        expect(path.length).toEqual(3);
        expect(path).toContain(i0);
        expect(path).toContain(i1);
        expect(path).toContain(n);
    });
    it('enqueues each node once even if there are multiple paths to it', function () {
        var t0 = new graph_1.Tensor([]);
        var n0 = new TestNode(g, '', {}, t0);
        var t1 = new graph_1.Tensor([]);
        var n1 = new TestNode(g, '', { 't0': t0 }, t1);
        var t2 = new graph_1.Tensor([]);
        var n2 = new TestNode(g, '', { 't0': t0 }, t2);
        var n3 = new TestNode(g, '', { 't1': t1, 't2': t2 }, new graph_1.Tensor([]));
        var set = graph_util.getUnorderedEvaluationSet([n3], []);
        expect(set.length).toEqual(4);
        expect(set).toContain(n0);
        expect(set).toContain(n1);
        expect(set).toContain(n2);
        expect(set).toContain(n3);
    });
});
describe('graph_util.getOrderedEvaluationSet', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('returns only node in unordered set', function () {
        var n = new TestNode(g, '', {}, new graph_1.Tensor([]));
        expect(graph_util.getOrderedEvaluationSet([n])).toEqual([n]);
    });
    it('orders dependencies first (2 nodes)', function () {
        var t0 = new graph_1.Tensor([]);
        var n0 = new TestNode(g, '', {}, t0);
        var n1 = new TestNode(g, '', { 't0': t0 }, new graph_1.Tensor([]));
        var unordered = [n1, n0];
        var ordered = [n0, n1];
        expect(graph_util.getOrderedEvaluationSet(unordered)).toEqual(ordered);
    });
    it('orders dependencies first (3 nodes)', function () {
        var t0 = new graph_1.Tensor([]);
        var n0 = new TestNode(g, '', {}, t0);
        var t1 = new graph_1.Tensor([]);
        var n1 = new TestNode(g, '', { 't0': t0 }, t1);
        var n2 = new TestNode(g, '', { 't0': t0, 't1': t1 }, new graph_1.Tensor([]));
        var unordered = [n1, n2, n0];
        var ordered = [n0, n1, n2];
        expect(graph_util.getOrderedEvaluationSet(unordered)).toEqual(ordered);
    });
    it('orders dependencies first (5 nodes)', function () {
        var t0 = new graph_1.Tensor([]);
        var n0 = new TestNode(g, '', {}, t0);
        var t1 = new graph_1.Tensor([]);
        var n1 = new TestNode(g, '', { 't0': t0 }, t1);
        var t2 = new graph_1.Tensor([]);
        var n2 = new TestNode(g, '', { 't0': t0 }, t2);
        var t3 = new graph_1.Tensor([]);
        var n3 = new TestNode(g, '', { 't1': t1, 't2': t2 }, t3);
        var n4 = new TestNode(g, '', { 't0': t0, 't3': t3 }, new graph_1.Tensor([]));
        var path = graph_util.getOrderedEvaluationSet([n4, n3, n2, n1, n0]);
        expect(path[0]).toBe(n0);
        var n2n1 = (path[1] === n2) && (path[2] === n1);
        var n1n2 = (path[1] === n1) && (path[2] === n2);
        expect(n2n1 || n1n2).toBe(true);
        expect(path[3]).toBe(n3);
        expect(path[4]).toBe(n4);
    });
});
describe('graph_util.isInputNode', function () {
    var g;
    var nda;
    beforeEach(function () {
        g = new graph_1.Graph();
        nda = dl.zeros([1]);
    });
    it('returns true for VariableNode', function () {
        expect(graph_util.isInputNode(new graph_1.VariableNode(g, '', nda))).toEqual(true);
    });
    it('returns true for PlaceholderNode', function () {
        expect(graph_util.isInputNode(new graph_1.PlaceholderNode(g, '', [1])))
            .toEqual(true);
    });
    it('returns true for ConstantNode', function () {
        expect(graph_util.isInputNode(new graph_1.ConstantNode(g, dl.zeros([1]))))
            .toEqual(true);
    });
    it('returns false for ReLUNode', function () {
        expect(graph_util.isInputNode(new graph_1.ReLUNode(g, new graph_1.Tensor([]))))
            .toEqual(false);
    });
});
describe('graph_util.isPassthroughNode', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('returns false for a node that produces new NDArray', function () {
        var x = g.placeholder('x', []);
        var node = new graph_1.SquareNode(g, x);
        var map = new tensor_array_map_1.TensorArrayMap();
        var xVal = ndarray_1.Scalar.new(3);
        map.set(x, xVal);
        var yVal = ndarray_1.Scalar.new(9);
        map.set(node.output, yVal);
        expect(graph_util.isPassthroughNode(node, map)).toBe(false);
        xVal.dispose();
        yVal.dispose();
    });
});
//# sourceMappingURL=graph_util_test.js.map