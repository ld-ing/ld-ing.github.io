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
var environment_1 = require("../environment");
var dl = require("../index");
var graph_1 = require("./graph");
var session_1 = require("./session");
var session_util = require("./session_util");
var tensor_array_map_1 = require("./tensor_array_map");
var TestNode = (function (_super) {
    __extends(TestNode, _super);
    function TestNode() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    TestNode.prototype.validate = function () { };
    return TestNode;
}(graph_1.Node));
describe('getTerminatingNodesFromFeedDictionary', function () {
    it('returns an empty node array from an empty FeedDictionary', function () {
        expect(session_util.getTerminatingNodesFromFeedDictionary(new session_1.FeedDictionary()))
            .toEqual([]);
    });
    it('returns the only node in the feed dictionary', function () {
        var math = environment_1.ENV.math;
        math.scope(function () {
            var node = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
            var fd = new session_1.FeedDictionary([{ tensor: node.output, data: dl.zeros([1]) }]);
            expect(session_util.getTerminatingNodesFromFeedDictionary(fd)).toEqual([
                node
            ]);
        });
    });
    it('returns every node from the feed dictionary', function () {
        var math = environment_1.ENV.math;
        math.scope(function () {
            var n0 = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
            var n1 = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
            var n2 = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
            var n3 = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
            var n4 = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
            var feeds = [
                { tensor: n0.output, data: dl.zeros([1]) },
                { tensor: n1.output, data: dl.zeros([1]) },
                { tensor: n2.output, data: dl.zeros([1]) },
                { tensor: n3.output, data: dl.zeros([1]) },
                { tensor: n4.output, data: dl.zeros([1]) }
            ];
            var fd = new session_1.FeedDictionary(feeds);
            var nodes = session_util.getTerminatingNodesFromFeedDictionary(fd);
            expect(nodes).toContain(n0);
            expect(nodes).toContain(n1);
            expect(nodes).toContain(n2);
            expect(nodes).toContain(n3);
            expect(nodes).toContain(n4);
        });
    });
});
describe('addPersistentArraysToTensorArrayMap', function () {
    var map;
    var g;
    var math = environment_1.ENV.math;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
        g = new graph_1.Graph();
    });
    it('does nothing with empty evaluationSet', function () {
        session_util.addPersistentArraysToTensorArrayMap([], map);
        expect(map.size()).toEqual(0);
    });
    it('adds the only VariableNode to the map', function () {
        var v = new graph_1.VariableNode(g, '', dl.zeros([1]));
        session_util.addPersistentArraysToTensorArrayMap([v], map);
        expect(map.get(v.output)).toBe(v.data);
    });
    it('adds the only ConstantNode to the map', function () {
        var c = new graph_1.ConstantNode(g, dl.zeros([1]));
        session_util.addPersistentArraysToTensorArrayMap([c], map);
        expect(map.get(c.output)).toBe(c.data);
    });
    it('does nothing with nodes that aren\'t VariableNodes or ConstantNodes', function () {
        var nodes = [new TestNode(g, '', {}, new graph_1.Tensor([]))];
        session_util.addPersistentArraysToTensorArrayMap(nodes, map);
        expect(map.size()).toEqual(0);
    });
    it('adds multiple VariableNodes to the map', function () {
        var nodes = [
            new graph_1.VariableNode(g, '', dl.zeros([1])),
            new graph_1.VariableNode(g, '', dl.zeros([1])),
            new graph_1.VariableNode(g, '', dl.zeros([1]))
        ];
        session_util.addPersistentArraysToTensorArrayMap(nodes, map);
        expect(map.get(nodes[0].output)).toBe(nodes[0].data);
        expect(map.get(nodes[1].output)).toBe(nodes[1].data);
        expect(map.get(nodes[2].output)).toBe(nodes[2].data);
    });
    it('adds multiple ConstantNodes to the map', function () {
        math.scope(function () {
            var nodes = [
                new graph_1.ConstantNode(g, dl.zeros([1])), new graph_1.ConstantNode(g, dl.zeros([1])),
                new graph_1.ConstantNode(g, dl.zeros([1]))
            ];
            session_util.addPersistentArraysToTensorArrayMap(nodes, map);
            expect(map.get(nodes[0].output)).toBe(nodes[0].data);
            expect(map.get(nodes[1].output)).toBe(nodes[1].data);
            expect(map.get(nodes[2].output)).toBe(nodes[2].data);
        });
    });
    it('skips non-VariableNode or ConstantNode entries in the set', function () {
        var nodes = [
            new TestNode(g, '', {}, new graph_1.Tensor([])),
            new graph_1.VariableNode(g, '', dl.zeros([1])),
            new TestNode(g, '', {}, new graph_1.Tensor([])),
            new graph_1.ConstantNode(g, dl.zeros([1])),
            new TestNode(g, '', {}, new graph_1.Tensor([])),
            new graph_1.VariableNode(g, '', dl.zeros([1]))
        ];
        session_util.addPersistentArraysToTensorArrayMap(nodes, map);
        expect(map.size()).toEqual(3);
        expect(map.get(nodes[1].output)).toBe(nodes[1].data);
        expect(map.get(nodes[3].output)).toBe(nodes[3].data);
        expect(map.get(nodes[5].output)).toBe(nodes[5].data);
    });
});
describe('loadInputsFromFeedDictionaryToTensorArrayMap', function () {
    var map;
    var math = environment_1.ENV.math;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
    });
    it('does nothing with empty feed dictionary', function () {
        var fd = new session_1.FeedDictionary();
        session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(0);
    });
    it('adds the only NDArray feed dict entry to the map', function () {
        var tensor = new graph_1.Tensor([1]);
        var fd = new session_1.FeedDictionary([{ tensor: tensor, data: dl.zeros([1]) }]);
        session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(1);
        expect(map.get(tensor)).toBe(fd.dict[tensor.id].data);
    });
    it('adds the only provider feed dict entry to the map', function () {
        var tensor = new graph_1.Tensor([2]);
        var ndarray = dl.zeros([2]);
        var provider = {
            getNextCopy: function () {
                return ndarray;
            },
            disposeCopy: function () { }
        };
        var fd = new session_1.FeedDictionary([{ tensor: tensor, data: provider }]);
        session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(1);
        expect(map.get(tensor)).toBe(ndarray);
    });
    it('adds every NDArray feed dict entry to the map', function () {
        var tensors = [
            new graph_1.Tensor([1]), new graph_1.Tensor([1]), new graph_1.Tensor([1]), new graph_1.Tensor([1]),
            new graph_1.Tensor([1])
        ];
        var feeds = tensors.map(function (tensor) {
            return { tensor: tensor, data: dl.zeros([1]) };
        });
        var fd = new session_1.FeedDictionary(feeds);
        session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(tensors.length);
        tensors.forEach(function (tensor) {
            return expect(map.get(tensor)).toBe(fd.dict[tensor.id].data);
        });
    });
    it('adds every provider feed dict entry to the map', function () {
        var tensors = [
            new graph_1.Tensor([1]), new graph_1.Tensor([1]), new graph_1.Tensor([1]), new graph_1.Tensor([1]),
            new graph_1.Tensor([1])
        ];
        var ndarrays = [];
        for (var i = 0; i < tensors.length; i++) {
            ndarrays.push(dl.zeros([1]));
        }
        var idx = 0;
        var provider = {
            getNextCopy: function () {
                var ndarray = ndarrays[idx];
                idx++;
                return ndarray;
            },
            disposeCopy: function () { }
        };
        var feeds = [];
        for (var i = 0; i < tensors.length; i++) {
            feeds.push({ tensor: tensors[i], data: provider });
        }
        var fd = new session_1.FeedDictionary(feeds);
        session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(tensors.length);
        for (var i = 0; i < tensors.length; i++) {
            expect(map.get(tensors[i])).toBe(ndarrays[i]);
        }
    });
    it('throws when provides data that does not match tensor shape', function () {
        var tensor = new graph_1.Tensor([4, 5]);
        var fd = new session_1.FeedDictionary([{ tensor: tensor, data: dl.zeros([2, 3]) }]);
        expect(function () { return session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math); })
            .toThrowError();
    });
});
describe('releaseFeedDictionaryInputsFromTensorArrayMap', function () {
    var map;
    var math = environment_1.ENV.math;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
    });
    it('doesn\'t remove anything when feed dictionary is empty', function () {
        map.set(new graph_1.Tensor([]), null);
        var fd = new session_1.FeedDictionary();
        session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(1);
    });
    it('doesn\'t remove tensors from map that don\'t exist in feed', function () {
        var fdTensor = new graph_1.Tensor([]);
        var nda = dl.zeros([1]);
        var fd = new session_1.FeedDictionary([{ tensor: fdTensor, data: dl.zeros([1]) }]);
        var nonFDTensor = new graph_1.Tensor([]);
        map.set(nonFDTensor, nda);
        session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(1);
        expect(map.get(nonFDTensor)).toBe(nda);
    });
    it('removes only tensor in map and feed dict', function () {
        var tensor = new graph_1.Tensor([]);
        var ndarray = dl.zeros([1]);
        var fd = new session_1.FeedDictionary([{ tensor: tensor, data: ndarray }]);
        map.set(tensor, ndarray);
        session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(0);
    });
    it('removes from map all tensors in feed dict', function () {
        var tensors = [new graph_1.Tensor([]), new graph_1.Tensor([]), new graph_1.Tensor([])];
        var feeds = tensors.map(function (tensor) {
            return { tensor: tensor, data: dl.zeros([1]) };
        });
        var fd = new session_1.FeedDictionary(feeds);
        tensors.forEach(function (tensor) { return map.set(tensor, fd.dict[tensor.id].data); });
        session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
        expect(map.size()).toEqual(0);
    });
});
describe('disposeAndInitializeOperationOutputs', function () {
    var map;
    var g;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
        g = new graph_1.Graph();
    });
    it('does nothing to map if set is empty', function () {
        session_util.disposeAndInitializeOperationOutputs([], map);
        expect(map.size()).toEqual(0);
    });
    it('does nothing to map if set has no input nodes', function () {
        var nodes = [
            new graph_1.VariableNode(g, '', dl.zeros([1])), new graph_1.PlaceholderNode(g, '', [1])
        ];
        session_util.disposeAndInitializeOperationOutputs(nodes, map);
        expect(map.size()).toEqual(0);
    });
    it('adds output tensor from only operation node', function () {
        var input = new graph_1.Tensor([]);
        var t = new graph_1.Tensor([]);
        session_util.disposeAndInitializeOperationOutputs([new TestNode(g, '', { 'in': input }, t)], map);
        expect(map.size()).toEqual(1);
        expect(map.hasNullArray(t)).toEqual(true);
    });
    it('adds output tensors from all operation nodes', function () {
        var input = new graph_1.Tensor([]);
        var tensors = [new graph_1.Tensor([]), new graph_1.Tensor([]), new graph_1.Tensor([])];
        var nodes = [];
        tensors.forEach(function (tensor) { return nodes.push(new TestNode(g, '', { 'in': input }, tensor)); });
        session_util.disposeAndInitializeOperationOutputs(nodes, map);
        expect(map.size()).toEqual(nodes.length);
        tensors.forEach(function (tensor) { return expect(map.hasNullArray(tensor)).toEqual(true); });
    });
});
describe('removeFeedDictionaryNodesFromEvaluationSet', function () {
    var set;
    beforeEach(function () {
        set = [];
    });
    it('does nothing when feed dictionary is empty', function () {
        var node = new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([]));
        set.push(node);
        var fd = new session_1.FeedDictionary();
        session_util.removeFeedDictionaryNodesFromEvaluationSet(fd, set);
        expect(set.length).toEqual(1);
        expect(set[0]).toBe(node);
    });
    it('removes only feed dict node from set', function () {
        set.push(new TestNode(new graph_1.Graph(), '', {}, new graph_1.Tensor([])));
        var fd = new session_1.FeedDictionary([{ tensor: set[0].output, data: dl.zeros([1]) }]);
        session_util.removeFeedDictionaryNodesFromEvaluationSet(fd, set);
        expect(set.length).toEqual(0);
    });
    it('removes only feed dict nodes from set', function () {
        var g = new graph_1.Graph();
        var remainingNodes = [
            new TestNode(g, '', {}, new graph_1.Tensor([])),
            new TestNode(g, '', {}, new graph_1.Tensor([])),
            new TestNode(g, '', {}, new graph_1.Tensor([]))
        ];
        set.push(remainingNodes[0]);
        set.push(new TestNode(g, '', {}, new graph_1.Tensor([])));
        var feeds = [];
        feeds.push({ tensor: set[set.length - 1].output, data: dl.zeros([1]) });
        set.push(remainingNodes[1]);
        set.push(new TestNode(g, '', {}, new graph_1.Tensor([])));
        feeds.push({ tensor: set[set.length - 1].output, data: dl.zeros([1]) });
        set.push(remainingNodes[2]);
        var fd = new session_1.FeedDictionary(feeds);
        session_util.removeFeedDictionaryNodesFromEvaluationSet(fd, set);
        expect(set).toEqual(remainingNodes);
    });
});
describe('throwErrorIfEvaluationSetContainsPlaceholderNodes', function () {
    var g;
    beforeEach(function () { return g = new graph_1.Graph(); });
    it('doesn\'t throw on an empty node array', function () {
        session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([]);
    });
    it('doesn\'t throw if array contains non-placeholder nodes', function () {
        session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([new TestNode(g, '', {}, new graph_1.Tensor([]))]);
    });
    it('throws if the array only contains a placeholder node', function () {
        expect(function () { return session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([new graph_1.PlaceholderNode(g, '', [])]); })
            .toThrowError(/Placeholder node/);
    });
    it('thrown error contains the tensor shape', function () {
        expect(function () { return session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([new graph_1.PlaceholderNode(g, '', [1, 2, 3, 4, 5])]); })
            .toThrowError(/[1, 2, 3, 4, 5]/);
    });
    it('throws if the non-first element in the array is a placeholder', function () {
        expect(function () { return session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([
            new TestNode(g, '', {}, new graph_1.Tensor([])),
            new graph_1.PlaceholderNode(g, '', [])
        ]); })
            .toThrowError(/Placeholder node/);
    });
});
//# sourceMappingURL=session_util_test.js.map