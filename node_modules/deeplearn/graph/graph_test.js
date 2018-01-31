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
var conv_util = require("../math/conv_util");
var graph_1 = require("./graph");
var session_1 = require("./session");
var session_util = require("./session_util");
var TestNode = (function (_super) {
    __extends(TestNode, _super);
    function TestNode() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    TestNode.prototype.validate = function () { };
    return TestNode;
}(graph_1.Node));
describe('Graph', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('nodes have ascending ids', function () {
        var a = new TestNode(g, '', {}, new graph_1.Tensor([]));
        var b = new TestNode(g, '', {}, new graph_1.Tensor([]));
        expect(b.id).toEqual(a.id + 1);
    });
    it('variable creates a node in the graph', function () {
        var v = g.variable('', dl.zeros([1]));
        expect(v.node.graph).toEqual(g);
    });
    it('variable creates a VariableNode in the graph', function () {
        var v = g.variable('', dl.zeros([1]));
        expect(v.node instanceof graph_1.VariableNode).toEqual(true);
    });
    it('variable passes name to graph node', function () {
        var v = g.variable('hello', dl.zeros([1]));
        expect(v.node.name).toEqual('hello');
    });
    it('mnist fully-connected', function () {
        var input = g.placeholder('input', [28 * 28]);
        var fc0W = g.variable('fc0W', dl.zeros([32, 28 * 28]));
        var fc0B = g.variable('fc0B', dl.zeros([32]));
        var fc0 = g.add(g.matmul(fc0W, input), fc0B);
        var relu0 = g.relu(fc0);
        var fc1W = g.variable('fc1W', dl.zeros([32, 32]));
        var fc1B = g.variable('fc1B', dl.zeros([32]));
        var fc1 = g.add(g.matmul(fc1W, relu0), fc1B);
        var relu1 = g.relu(fc1);
        var fc2W = g.variable('fc2W', dl.zeros([32, 32]));
        var fc2B = g.variable('fc2B', dl.zeros([32]));
        var fc2 = g.add(g.matmul(fc2W, relu1), fc2B);
        var relu2 = g.relu(fc2);
        var fc3W = g.variable('fc3W', dl.zeros([10, 32]));
        var fc3B = g.variable('fc3B', dl.zeros([10]));
        var fc3 = g.add(g.matmul(fc3W, relu2), fc3B);
        var fd = new session_1.FeedDictionary([{ tensor: input, data: dl.zeros([1]) }]);
        var orderedEvaluationSet = session_util.getOrderedEvaluationSetFromEvalTensor([fc3], fd);
        expect(orderedEvaluationSet.length).toBeGreaterThan(1);
    });
});
describe('Variable validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('null data throws', function () {
        expect(function () { return g.variable('test', null); }).toThrowError();
    });
    it('non null data does not throw', function () {
        g.variable('test', dl.zeros([5]));
    });
});
describe('Placeholder validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('does not throw', function () {
        expect(g.placeholder('test', [1, 2, 3]).shape).toEqual([1, 2, 3]);
    });
});
describe('Constant', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('null data throws', function () {
        expect(function () { return g.constant(null); }).toThrowError();
    });
    it('non null data does not throw', function () {
        expect(g.constant(dl.zeros([5])).shape).toEqual([5]);
    });
    it('from a single value', function () {
        var c = g.constant(3);
        expect(c.shape).toEqual([]);
        var values = c.node.data.dataSync();
        expect(values).toEqual(new Float32Array([3]));
    });
    it('from 1d array', function () {
        var c = g.constant([1, 2, 3]);
        expect(c.shape).toEqual([3]);
        var values = c.node.data.dataSync();
        expect(values).toEqual(new Float32Array([1, 2, 3]));
    });
    it('from 2d array', function () {
        var c = g.constant([[1, 2, 3], [4, 5, 6]]);
        expect(c.shape).toEqual([2, 3]);
        var values = c.node.data.dataSync();
        expect(values).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('from 3d array', function () {
        var c = g.constant([[[1], [2], [3]], [[4], [5], [6]]]);
        expect(c.shape).toEqual([2, 3, 1]);
        var values = c.node.data.dataSync();
        expect(values).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('from 4d array', function () {
        var c = g.constant([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]);
        expect(c.shape).toEqual([2, 3, 1, 1]);
        var values = c.node.data.dataSync();
        expect(values).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });
});
describe('Reshape validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different sizes throws', function () {
        expect(function () { return g.reshape(new graph_1.Tensor([5, 4]), [3, 3]); }).toThrowError();
    });
    it('Same size does not throw', function () {
        expect(g.reshape(new graph_1.Tensor([5, 4]), [20]).shape).toEqual([20]);
    });
});
describe('FusedLinearCombination validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different shape tensors throws', function () {
        expect(function () { return g.fusedLinearCombination(new graph_1.Tensor([3, 4]), new graph_1.Tensor([1]), new graph_1.Tensor([]), new graph_1.Tensor([])); })
            .toThrowError();
    });
    it('Non scalar c1 throws', function () {
        expect(function () { return g.fusedLinearCombination(new graph_1.Tensor([3, 4]), new graph_1.Tensor([3, 4]), new graph_1.Tensor([1, 2]), new graph_1.Tensor([])); })
            .toThrowError();
    });
    it('Non scalar c2 throws', function () {
        expect(function () { return g.fusedLinearCombination(new graph_1.Tensor([3, 4]), new graph_1.Tensor([3, 4]), new graph_1.Tensor([]), new graph_1.Tensor([1, 2])); })
            .toThrowError();
    });
    it('does not throw when shapes correct', function () {
        expect(g.fusedLinearCombination(new graph_1.Tensor([3, 4]), new graph_1.Tensor([3, 4]), new graph_1.Tensor([]), new graph_1.Tensor([]))
            .shape)
            .toEqual([3, 4]);
    });
});
describe('Add validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different shapes throws', function () {
        expect(function () { return g.add(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3])); })
            .toThrowError();
    });
    it('Same size does not throw', function () {
        expect(g.add(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
    it('1D broadcasted to 2D does not throw', function () {
        expect(g.add(new graph_1.Tensor([5, 3]), new graph_1.Tensor([3])).shape).toEqual([5, 3]);
    });
    it('Another 1D broadcasted to 2D does not throw', function () {
        expect(g.add(new graph_1.Tensor([3]), new graph_1.Tensor([7, 3])).shape).toEqual([7, 3]);
    });
    it('Non-matching broadcast throws', function () {
        expect(function () { return g.add(new graph_1.Tensor([5, 3]), new graph_1.Tensor([5])); }).toThrowError();
    });
});
describe('Subtract validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different shapes throws', function () {
        expect(function () { return g.subtract(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3])); })
            .toThrowError();
    });
    it('Same size does not throw', function () {
        expect(g.subtract(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape).toEqual([
            5, 4
        ]);
    });
});
describe('Multiply validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different shapes throws', function () {
        expect(function () { return g.multiply(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3])); })
            .toThrowError();
    });
    it('Same size does not throw', function () {
        expect(g.multiply(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape).toEqual([
            5, 4
        ]);
    });
});
describe('Divide validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different shapes throws', function () {
        expect(function () { return g.divide(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3])); })
            .toThrowError();
    });
    it('Same size does not throw', function () {
        expect(g.divide(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape).toEqual([
            5, 4
        ]);
    });
});
describe('Reduce sum validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('does not throw', function () {
        expect(g.reduceSum(new graph_1.Tensor([5, 4, 4, 9])).shape).toEqual([]);
    });
});
describe('Concat1d validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Non 1-rank tensor x1 throws', function () {
        expect(function () { return g.concat1d(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1])); })
            .toThrowError();
    });
    it('Non 1-rank tensor x2 throws', function () {
        expect(function () { return g.concat1d(new graph_1.Tensor([5]), new graph_1.Tensor([1, 2])).shape; })
            .toThrowError();
    });
    it('Axis=0 shapes the same does not throw', function () {
        expect(g.concat1d(new graph_1.Tensor([5]), new graph_1.Tensor([1])).shape).toEqual([6]);
    });
});
describe('Concat2d validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Non 2-rank tensor x1 throws', function () {
        expect(function () { return g.concat2d(new graph_1.Tensor([5]), new graph_1.Tensor([1, 2]), 0); })
            .toThrowError();
    });
    it('Non 2-rank tensor x2 throws', function () {
        expect(function () { return g.concat2d(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1]), 0); })
            .toThrowError();
    });
    it('Axis=0 different shapes throw', function () {
        expect(function () { return g.concat2d(new graph_1.Tensor([2, 3]), new graph_1.Tensor([4, 4]), 0); })
            .toThrowError();
    });
    it('Axis=0 shapes the same doe not throw', function () {
        expect(g.concat2d(new graph_1.Tensor([2, 3]), new graph_1.Tensor([4, 3]), 0).shape)
            .toEqual([6, 3]);
    });
    it('Axis=1 different shapes throw', function () {
        expect(function () { return g.concat2d(new graph_1.Tensor([2, 3]), new graph_1.Tensor([4, 4]), 1); })
            .toThrowError();
    });
    it('Axis=1 shapes the same doe not throw', function () {
        expect(g.concat2d(new graph_1.Tensor([2, 4]), new graph_1.Tensor([2, 3]), 1).shape)
            .toEqual([2, 7]);
    });
});
describe('Concat3d validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Non 3-rank tensor x1 throws', function () {
        expect(function () { return g.concat3d(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3]), 0); })
            .toThrowError();
    });
    it('Non 3-rank tensor x2 throws', function () {
        expect(function () { return g.concat3d(new graph_1.Tensor([5, 4, 1]), new graph_1.Tensor([1, 2]), 0); })
            .toThrowError();
    });
    it('Axis=0 different shapes throws', function () {
        expect(function () { return g.concat3d(new graph_1.Tensor([5, 4, 1]), new graph_1.Tensor([1, 2, 1]), 0); })
            .toThrowError();
    });
    it('Axis=1 different shapes throws', function () {
        expect(function () { return g.concat3d(new graph_1.Tensor([5, 4, 1]), new graph_1.Tensor([1, 2, 1]), 1); })
            .toThrowError();
    });
    it('Axis=2 different shapes throws', function () {
        expect(function () { return g.concat3d(new graph_1.Tensor([5, 4, 1]), new graph_1.Tensor([1, 2, 1]), 2); })
            .toThrowError();
    });
    it('Axis=0 shapes the same does not throw', function () {
        expect(g.concat3d(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([1, 4, 3]), 0).shape)
            .toEqual([6, 4, 3]);
    });
    it('Axis=1 shapes the same does not throw', function () {
        expect(g.concat3d(new graph_1.Tensor([5, 3, 3]), new graph_1.Tensor([5, 4, 3]), 1).shape)
            .toEqual([5, 7, 3]);
    });
    it('Axis=2 shapes the same does not throw', function () {
        expect(g.concat3d(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([5, 4, 1]), 2).shape)
            .toEqual([5, 4, 4]);
    });
});
describe('Concat4d validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Non 4-rank tensor x1 throws', function () {
        expect(function () { return g.concat4d(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3, 4]), 0); })
            .toThrowError();
    });
    it('Non 4-rank tensor x2 throws', function () {
        expect(function () { return g.concat4d(new graph_1.Tensor([5, 4, 1]), new graph_1.Tensor([1, 2, 3, 4]), 0); })
            .toThrowError();
    });
    it('Axis=0 different shapes throws', function () {
        expect(function () { return g.concat4d(new graph_1.Tensor([5, 4, 1, 1]), new graph_1.Tensor([1, 2, 1, 1]), 0); })
            .toThrowError();
    });
    it('Axis=1 different shapes throws', function () {
        expect(function () { return g.concat4d(new graph_1.Tensor([5, 4, 1, 1]), new graph_1.Tensor([1, 2, 1, 1]), 1); })
            .toThrowError();
    });
    it('Axis=2 different shapes throws', function () {
        expect(function () { return g.concat4d(new graph_1.Tensor([5, 4, 1, 1]), new graph_1.Tensor([1, 2, 1, 1]), 2); })
            .toThrowError();
    });
    it('Axis=3 different shapes throws', function () {
        expect(function () { return g.concat4d(new graph_1.Tensor([5, 4, 1, 1]), new graph_1.Tensor([1, 2, 1, 1]), 3); })
            .toThrowError();
    });
    it('Axis=0 shapes the same does not throw', function () {
        expect(g.concat4d(new graph_1.Tensor([5, 4, 3, 1]), new graph_1.Tensor([1, 4, 3, 1]), 0).shape)
            .toEqual([6, 4, 3, 1]);
    });
    it('Axis=1 shapes the same does not throw', function () {
        expect(g.concat4d(new graph_1.Tensor([5, 3, 3, 1]), new graph_1.Tensor([5, 4, 3, 1]), 1).shape)
            .toEqual([5, 7, 3, 1]);
    });
    it('Axis=2 shapes the same does not throw', function () {
        expect(g.concat4d(new graph_1.Tensor([5, 4, 3, 1]), new graph_1.Tensor([5, 4, 1, 1]), 2).shape)
            .toEqual([5, 4, 4, 1]);
    });
    it('Axis=3 shapes the same does not throw', function () {
        expect(g.concat4d(new graph_1.Tensor([5, 4, 3, 1]), new graph_1.Tensor([5, 4, 3, 2]), 3).shape)
            .toEqual([5, 4, 3, 3]);
    });
});
describe('matmul validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Wrong rank x1 throws', function () {
        expect(function () { return g.matmul(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([1, 2])); })
            .toThrowError();
    });
    it('Wrong rank x2 throws', function () {
        expect(function () { return g.matmul(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3])); })
            .toThrowError();
    });
    it('Inner dimensions of matrix multiply do not match throws', function () {
        expect(function () { return g.matmul(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 5])); })
            .toThrowError();
    });
    it('Inner dimensions of matrix times vector does not match throws', function () {
        expect(function () { return g.matmul(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5])); }).toThrowError();
    });
    it('Inner dimensions of vector times matrix does not match throws', function () {
        expect(function () { return g.matmul(new graph_1.Tensor([5]), new graph_1.Tensor([4, 5])); }).toThrowError();
    });
    it('Vector times vector shapes dont match throws', function () {
        expect(function () { return g.matmul(new graph_1.Tensor([5]), new graph_1.Tensor([4])); }).toThrowError();
    });
    it('Matrix times matrix inner dimensions match does not throw', function () {
        expect(g.matmul(new graph_1.Tensor([5, 4]), new graph_1.Tensor([4, 6])).shape).toEqual([
            5, 6
        ]);
    });
    it('Vector times matrix inner dimensions match does not throw', function () {
        expect(g.matmul(new graph_1.Tensor([4]), new graph_1.Tensor([4, 6])).shape).toEqual([6]);
    });
    it('Matrix times vector inner dimensions match does not throw', function () {
        expect(g.matmul(new graph_1.Tensor([4, 6]), new graph_1.Tensor([6])).shape).toEqual([4]);
    });
});
describe('conv2d validation', function () {
    var g;
    var fieldSize;
    var outputDepth;
    var stride;
    var zeroPad;
    beforeEach(function () {
        g = new graph_1.Graph();
        fieldSize = 4;
        outputDepth = 10;
        stride = 1;
        zeroPad = 1;
    });
    it('Wrong rank x throws', function () {
        expect(function () { return g.conv2d(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3, 4]), new graph_1.Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad); })
            .toThrowError();
    });
    it('Wrong rank weights throws', function () {
        expect(function () { return g.conv2d(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([1, 2, 3]), new graph_1.Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad); })
            .toThrowError();
    });
    it('Wrong rank biases throws', function () {
        expect(function () { return g.conv2d(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([1, 2, 3, 4]), new graph_1.Tensor([5, 5]), fieldSize, outputDepth, stride, zeroPad); })
            .toThrowError();
    });
    it('Input depths dont match throws', function () {
        expect(function () { return g.conv2d(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([1, 2, 100, 4]), new graph_1.Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad); })
            .toThrowError();
    });
    it('Shapes matches does not throw', function () {
        var expectedShape = conv_util.computeOutputShape3D([5, 4, 3], fieldSize, outputDepth, stride, zeroPad);
        expect(g.conv2d(new graph_1.Tensor([5, 4, 3]), new graph_1.Tensor([1, 2, 3, 4]), new graph_1.Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad)
            .shape)
            .toEqual(expectedShape);
    });
});
describe('maxpool validation', function () {
    var g;
    var fieldSize;
    var stride;
    var zeroPad;
    beforeEach(function () {
        g = new graph_1.Graph();
        fieldSize = 4;
        stride = 1;
        zeroPad = 1;
    });
    it('Wrong rank x throws', function () {
        expect(function () { return g.maxPool(new graph_1.Tensor([5, 4]), fieldSize, stride, zeroPad); })
            .toThrowError();
    });
    it('Shapes matches does not throw', function () {
        var expectedShape = conv_util.computeOutputShape3D([5, 4, 3], fieldSize, 3, stride, zeroPad);
        expect(g.maxPool(new graph_1.Tensor([5, 4, 3]), fieldSize, stride, zeroPad).shape)
            .toEqual(expectedShape);
    });
});
describe('relu validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.relu(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('leakyRelu validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.leakyRelu(new graph_1.Tensor([5, 4]), 0.2).shape).toEqual([5, 4]);
    });
});
describe('pRelu validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Different shapes throws', function () {
        expect(function () { return g.prelu(new graph_1.Tensor([5, 4]), new graph_1.Tensor([1, 2, 3])); })
            .toThrowError();
    });
    it('Same size does not throw', function () {
        expect(g.prelu(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape).toEqual([
            5, 4
        ]);
    });
});
describe('elu validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.elu(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('exp validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.exp(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('log validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.log(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('tanh validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.tanh(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('sigmoid validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.sigmoid(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('square validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Does not throw', function () {
        expect(g.square(new graph_1.Tensor([5, 4])).shape).toEqual([5, 4]);
    });
});
describe('softmaxCrossEntropy validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Shapes not equal throws', function () {
        expect(function () { return g.softmaxCrossEntropyCost(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4, 3])); })
            .toThrowError();
    });
    it('Does not throw', function () {
        expect(g.softmaxCrossEntropyCost(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape)
            .toEqual([]);
    });
});
describe('meanSquaredCost validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Shapes not equal throws', function () {
        expect(function () { return g.meanSquaredCost(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4, 3])); })
            .toThrowError();
    });
    it('Does not throw', function () {
        expect(g.meanSquaredCost(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape)
            .toEqual([]);
    });
});
describe('argmaxEquals validation', function () {
    var g;
    beforeEach(function () {
        g = new graph_1.Graph();
    });
    it('Shapes not equal throws', function () {
        expect(function () { return g.argmaxEquals(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4, 3])); })
            .toThrowError();
    });
    it('Does not throw', function () {
        expect(g.argmaxEquals(new graph_1.Tensor([5, 4]), new graph_1.Tensor([5, 4])).shape)
            .toEqual([1]);
    });
});
describe('Tensor', function () {
    it('captures shape from constructor', function () {
        var t = new graph_1.Tensor([1, 2, 3, 4]);
        expect(t.shape).toEqual([1, 2, 3, 4]);
    });
    it('has unique ascending ids', function () {
        var a = new graph_1.Tensor([]);
        var b = new graph_1.Tensor([]);
        expect(b.id).toEqual(a.id + 1);
    });
});
//# sourceMappingURL=graph_test.js.map