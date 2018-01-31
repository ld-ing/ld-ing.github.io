"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var dl = require("../index");
var math_1 = require("../math/math");
var ndarray_1 = require("../math/ndarray");
var sgd_optimizer_1 = require("../math/optimizers/sgd_optimizer");
var test_util = require("../test_util");
var graph_1 = require("./graph");
var session_1 = require("./session");
describe('FeedDictionary', function () {
    it('ctor leaves dict empty if no args are passed', function () {
        expect(Object.keys(new session_1.FeedDictionary().dict).length).toEqual(0);
    });
    it('ctor populates dict from only feed entry', function () {
        var math = environment_1.ENV.math;
        math.scope(function () {
            var e = { tensor: new graph_1.Tensor([]), data: dl.zeros([1]) };
            var d = new session_1.FeedDictionary([e]);
            expect(Object.keys(d.dict).length).toEqual(1);
            expect(d.dict[e.tensor.id]).toBe(e);
        });
    });
    it('ctor populates dict from many entries', function () {
        var entries = [
            { tensor: new graph_1.Tensor([]), data: dl.zeros([1]) },
            { tensor: new graph_1.Tensor([]), data: dl.zeros([1]) },
            { tensor: new graph_1.Tensor([]), data: dl.zeros([1]) },
            { tensor: new graph_1.Tensor([]), data: dl.zeros([1]) }
        ];
        var d = new session_1.FeedDictionary(entries);
        expect(Object.keys(d.dict).length).toEqual(entries.length);
        entries.forEach(function (entry) { return expect(d.dict[entry.tensor.id]).toBe(entry); });
    });
    it('add adds entry to map keyed on tensor id', function () {
        var t = new graph_1.Tensor([]);
        var nda = dl.zeros([1]);
        var fd = new session_1.FeedDictionary([{ tensor: t, data: nda }]);
        expect(fd.dict[t.id].tensor).toBe(t);
        expect(fd.dict[t.id].data).toBe(nda);
    });
});
describe('Session', function () {
    var g;
    beforeEach(function () { return g = new graph_1.Graph(); });
    it('mnist fc', function () {
        var math = environment_1.ENV.math;
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
        var session = new session_1.Session(g, math);
        session.eval(fc3, [{ tensor: input, data: dl.zeros([28 * 28]) }]);
    });
    it('y=x^2 + 3: CPU', function () {
        var x = g.placeholder('x', [2]);
        var y = g.add(g.square(x), g.constant(3));
        var session = new session_1.Session(g, environment_1.ENV.math);
        var yVal = session.eval(y, [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
        var expected = new Float32Array([28, 19]);
        test_util.expectArraysClose(yVal.dataSync(), expected);
    });
    it('y=x^2 + 3: GPU', function () {
        var math = environment_1.ENV.math;
        var x = g.placeholder('x', [2]);
        var y = g.add(g.square(x), g.constant(3));
        var session = new session_1.Session(g, math);
        math.scope(function () {
            var yVal = session.eval(y, [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
            var expected = new Float32Array([28, 19]);
            test_util.expectArraysClose(yVal.dataSync(), expected);
        });
    });
    it('Non-placeholder feed: y=x^2 + 3 (feed x^2)', function () {
        var math = environment_1.ENV.math;
        var x = g.placeholder('x', [2]);
        var xSquared = g.square(x);
        var y = g.add(xSquared, g.constant(3));
        var session = new session_1.Session(g, math);
        math.scope(function () {
            var yVal = session.eval(y, [{ tensor: xSquared, data: ndarray_1.Array1D.new([25, 16]) }]);
            var expected = new Float32Array([28, 19]);
            test_util.expectArraysClose(yVal.dataSync(), expected);
        });
    });
    it('Eval multiple tensors that share graph: y=x^2 + 3, z=x^2 + 2', function () {
        var math = environment_1.ENV.math;
        var x = g.placeholder('x', [2]);
        var xSquared = g.square(x);
        var y = g.add(xSquared, g.constant(3));
        var z = g.add(xSquared, g.constant(2));
        var session = new session_1.Session(g, math);
        math.scope(function () {
            var result = session.evalAll([y, z], [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
            var expectedY = new Float32Array([28, 19]);
            var expectedZ = new Float32Array([27, 18]);
            test_util.expectArraysClose(result[0].dataSync(), expectedY);
            test_util.expectArraysClose(result[1].dataSync(), expectedZ);
        });
    });
    it('Eval 2 tensors that share a split graph: y=x^2 + x, z=y + 1', function () {
        var math = environment_1.ENV.math;
        var x = g.placeholder('x', [2]);
        var xSquared = g.square(x);
        var y = g.add(xSquared, x);
        var z = g.add(y, g.constant(1));
        var session = new session_1.Session(g, math);
        math.scope(function () {
            var result1 = session.eval(y, [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
            var expectedY = new Float32Array([30, 20]);
            test_util.expectArraysClose(result1.dataSync(), expectedY);
            var result2 = session.eval(z, [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
            var expectedZ = new Float32Array([31, 21]);
            test_util.expectArraysClose(result2.dataSync(), expectedZ);
        });
    });
    it('Backprop through a  with 2 outputs, input is scalar', function () {
        var math = environment_1.ENV.math;
        var two = ndarray_1.Scalar.new(2);
        var one = ndarray_1.Scalar.new(1);
        var negOne = ndarray_1.Scalar.new(-1);
        var x = g.placeholder('x', []);
        var y = g.square(x);
        var z = g.add(x, g.constant(3));
        var w = g.add(y, z);
        var optimizer = new sgd_optimizer_1.SGDOptimizer(0.1);
        var session = new session_1.Session(g, math);
        var idx = 0;
        var xs = [two, one, negOne];
        var inputProvider = {
            getNextCopy: function () {
                return xs[idx++];
            },
            disposeCopy: function (math, example) { }
        };
        session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
        var dwdx = session.gradientArrayMap.get(x).get();
        expect(dwdx).toBe(5);
        session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
        dwdx = session.gradientArrayMap.get(x).get();
        expect(dwdx).toBe(3);
        session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
        dwdx = session.gradientArrayMap.get(x).get();
        expect(dwdx).toBe(-1);
    });
    it('Backprop through a node with 2 outputs, input is Array1D', function () {
        var math = environment_1.ENV.math;
        var x = g.placeholder('x', [2]);
        var y = g.square(x);
        var z = g.add(x, g.constant(3));
        var w = g.reduceSum(g.add(y, z));
        var optimizer = new sgd_optimizer_1.SGDOptimizer(0.1);
        var session = new session_1.Session(g, math);
        var inputProvider = {
            getNextCopy: function () {
                return ndarray_1.Array1D.new([2, 4]);
            },
            disposeCopy: function (math, example) { }
        };
        session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
        var dwdx = session.gradientArrayMap.get(x).dataSync();
        test_util.expectArraysClose(dwdx, new Float32Array([5, 9]));
    });
    it('Specify which variables to update (var_list)', function () {
        var math = environment_1.ENV.math;
        var x = g.placeholder('x', [2]);
        var b0 = g.variable('b0', dl.zeros([2]));
        var p = g.add(x, b0);
        var q = g.square(p);
        var b1 = g.variable('b1', dl.zeros([2]));
        var r = g.add(q, b1);
        var yPrediction = g.reduceSum(r);
        var yTrue = g.constant(1);
        var cost = g.meanSquaredCost(yTrue, yPrediction);
        var session = new session_1.Session(g, math);
        var inputProvider = {
            getNextCopy: function () {
                return ndarray_1.Array1D.new([1, 2]);
            },
            disposeCopy: function (math, example) { }
        };
        var optimizerOnlyB0 = new sgd_optimizer_1.SGDOptimizer(0.1, [b0.node]);
        session.train(cost, [{ tensor: x, data: inputProvider }], 2, optimizerOnlyB0, undefined);
        var b0After1 = session.activationArrayMap.get(b0).dataSync();
        var b1After1 = session.activationArrayMap.get(b1).dataSync();
        test_util.expectArraysClose(b0After1, new Float32Array([-0.8, -1.6]));
        test_util.expectArraysClose(b1After1, new Float32Array([0, 0]));
        var optimizerAll = new sgd_optimizer_1.SGDOptimizer(0.1);
        session.train(cost, [{ tensor: x, data: inputProvider }], 2, optimizerAll, undefined);
        var b0After2 = session.activationArrayMap.get(b0).dataSync();
        var b1After2 = session.activationArrayMap.get(b1).dataSync();
        expect(b0After2 === b0After1).toEqual(false);
        expect(b1After2 === b1After1).toEqual(false);
    });
    it('Safe mode math, no math scope eval throws', function () {
        var safeMode = true;
        var math = new math_1.NDArrayMath('webgl', safeMode);
        environment_1.ENV.setMath(math);
        expect(function () {
            var x = g.placeholder('x', [2]);
            var y = g.square(x);
            var session = new session_1.Session(g, math);
            session.eval(y, [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
        }).toThrowError();
        environment_1.ENV.reset();
    });
    it('Safe mode math, math scope eval does not throw', function () {
        var safeMode = true;
        var math = new math_1.NDArrayMath('webgl', safeMode);
        environment_1.ENV.setMath(math);
        math.scope(function () {
            var x = g.placeholder('x', [2]);
            var y = g.square(x);
            var session = new session_1.Session(g, math);
            var yVal = session.eval(y, [{ tensor: x, data: ndarray_1.Array1D.new([5, 4]) }]);
            var expected = new Float32Array([25, 16]);
            test_util.expectArraysClose(yVal.dataSync(), expected);
        });
        environment_1.ENV.reset();
    });
    it('Safe mode math, math scope train does not throw', function () {
        var safeMode = true;
        var math = new math_1.NDArrayMath('webgl', safeMode);
        environment_1.ENV.setMath(math);
        var inputProvider = {
            getNextCopy: function () {
                return ndarray_1.Array1D.new([2, 4]);
            },
            disposeCopy: function (math, example) { }
        };
        math.scope(function () {
            var optimizer = new sgd_optimizer_1.SGDOptimizer(0.1);
            var session = new session_1.Session(g, math);
            var x = g.placeholder('x', [2]);
            var y = g.square(x);
            var z = g.add(x, g.constant(3));
            var w = g.reduceSum(g.add(y, z));
            session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
            var dwdx = session.gradientArrayMap.get(x).dataSync();
            test_util.expectArraysClose(dwdx, new Float32Array([5, 9]));
        });
        environment_1.ENV.reset();
    });
    it('Safe mode math, no math scope train throws', function () {
        var safeMode = true;
        var math = new math_1.NDArrayMath('webgl', safeMode);
        environment_1.ENV.setMath(math);
        var inputProvider = {
            getNextCopy: function () {
                return ndarray_1.Array1D.new([2, 4]);
            },
            disposeCopy: function (math, example) { }
        };
        expect(function () {
            var session = new session_1.Session(g, math);
            var optimizer = new sgd_optimizer_1.SGDOptimizer(0.1);
            var x = g.placeholder('x', [2]);
            var y = g.square(x);
            var z = g.add(x, g.constant(3));
            var w = g.reduceSum(g.add(y, z));
            session.train(w, [{ tensor: x, data: inputProvider }], 1, optimizer);
        }).toThrowError();
        environment_1.ENV.reset();
    });
});
//# sourceMappingURL=session_test.js.map