"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../../test_util");
var ndarray_1 = require("../ndarray");
var tape_util = require("./tape_util");
{
    var tests = function (it) {
        it('getFilteredNodesXToY no paths from x to y', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var intermediate1 = ndarray_1.Scalar.new(0);
            var intermediate2 = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(2);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: intermediate1,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate2: intermediate2 },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);
            expect(filteredTapeNodes.length).toBe(0);
            expect(filteredTapeNodes).toEqual([]);
        });
        it('getFilteredNodesXToY one operation x => y', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: null
                }];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);
            expect(filteredTapeNodes.length).toBe(1);
            expect(filteredTapeNodes).toEqual(tape);
        });
        it('getFilteredNodesXToY 1 operation [x0, x1] => y, all input paths', function (math) {
            var x0 = ndarray_1.Scalar.new(0);
            var x1 = ndarray_1.Scalar.new(1);
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x0: x0, x1: x1 },
                    },
                    output: y,
                    gradient: null
                }];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x0, x1], y);
            expect(filteredTapeNodes.length).toBe(1);
            expect(filteredTapeNodes).toEqual(tape);
        });
        it('getFilteredNodesXToY one operation [x0, x1] => y, one input paths', function (math) {
            var x0 = ndarray_1.Scalar.new(0);
            var x1 = ndarray_1.Scalar.new(1);
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x0: x0, x1: x1 },
                    },
                    output: y,
                    gradient: null
                }];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x0], y);
            expect(filteredTapeNodes.length).toBe(1);
            expect(filteredTapeNodes[0]).toEqual({
                id: 0,
                type: 'kernel',
                name: 'node0',
                inputAndArgs: {
                    inputs: { x0: x0 },
                },
                output: y,
                gradient: null
            });
        });
        it('getFilteredNodesXToY two operations x => intermediate => y', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var intermediate = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(2);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: intermediate,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate: intermediate },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);
            expect(filteredTapeNodes.length).toBe(2);
            expect(filteredTapeNodes).toEqual(tape);
        });
        it('getFilteredNodesXToY two operations [x0, x1], [x2] => ' +
            'intermediate => y', function (math) {
            var x0 = ndarray_1.Scalar.new(1);
            var x1 = ndarray_1.Scalar.new(2);
            var x2 = ndarray_1.Scalar.new(3);
            var intermediate = ndarray_1.Scalar.new(4);
            var y = ndarray_1.Scalar.new(2);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x0: x0, x1: x1 },
                    },
                    output: intermediate,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { x2: x2, intermediate: intermediate },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x0, x1, x2], y);
            expect(filteredTapeNodes.length).toBe(2);
            expect(filteredTapeNodes).toEqual(tape);
        });
        it('getFilteredNodesXToY x => y and x => orphan', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var orphan = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(2);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: orphan,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);
            expect(filteredTapeNodes.length).toBe(1);
            expect(filteredTapeNodes[0]).toEqual(tape[1]);
        });
        it('getFilteredNodesXToY x => y and orphan => y', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var orphan = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x, orphan: orphan },
                    },
                    output: y,
                    gradient: null
                }];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);
            expect(filteredTapeNodes.length).toBe(1);
            expect(filteredTapeNodes[0]).toEqual({
                id: 0,
                type: 'kernel',
                name: 'node0',
                inputAndArgs: {
                    inputs: { x: x },
                },
                output: y,
                gradient: null
            });
        });
        it('getFilteredNodesXToY x => {intermediate, orphan1} and ' +
            '{orphan2, intermediate} => {y, orphan3}', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var intermediate = ndarray_1.Scalar.new(5);
            var orphan1 = ndarray_1.Scalar.new(1);
            var orphan2 = ndarray_1.Scalar.new(2);
            var orphan3 = ndarray_1.Scalar.new(3);
            var y = ndarray_1.Scalar.new(2);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: { orphan1: orphan1, intermediate: intermediate },
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate: intermediate, orphan2: orphan2 },
                    },
                    output: { y: y, orphan3: orphan3 },
                    gradient: null
                }
            ];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x], y);
            expect(filteredTapeNodes.length).toBe(2);
            expect(filteredTapeNodes[0]).toEqual({
                id: 0,
                type: 'kernel',
                name: 'node0',
                inputAndArgs: {
                    inputs: { x: x },
                },
                output: { intermediate: intermediate },
                gradient: null
            });
            expect(filteredTapeNodes[1]).toEqual({
                id: 1,
                type: 'kernel',
                name: 'node1',
                inputAndArgs: {
                    inputs: { intermediate: intermediate },
                },
                output: { y: y },
                gradient: null
            });
        });
        it('getFilteredNodesXToY x0 => orphan0, ' +
            'x0 => intermediate0, x0 => intermediate1, ' +
            '[intermediate0, intermediate1, x1, orphan1] => {y, orphan2}', function (math) {
            var x0 = ndarray_1.Scalar.new(1);
            var orphan0 = ndarray_1.Scalar.new(2);
            var intermediate0 = ndarray_1.Scalar.new(3);
            var intermediate1 = ndarray_1.Scalar.new(4);
            var x1 = ndarray_1.Scalar.new(5);
            var orphan1 = ndarray_1.Scalar.new(6);
            var y = ndarray_1.Scalar.new(7);
            var orphan2 = ndarray_1.Scalar.new(8);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x0: x0 },
                    },
                    output: intermediate0,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { x0: x0 },
                    },
                    output: intermediate1,
                    gradient: null
                },
                {
                    id: 2,
                    type: 'kernel',
                    name: 'node2',
                    inputAndArgs: {
                        inputs: { x0: x0 },
                    },
                    output: orphan0,
                    gradient: null
                },
                {
                    id: 3,
                    type: 'kernel',
                    name: 'node3',
                    inputAndArgs: {
                        inputs: { intermediate0: intermediate0, intermediate1: intermediate1, x1: x1, orphan1: orphan1 },
                    },
                    output: { y: y, orphan2: orphan2 },
                    gradient: null
                }
            ];
            var filteredTapeNodes = tape_util.getFilteredNodesXToY(tape, [x0, x1], y);
            expect(filteredTapeNodes.length).toBe(3);
            expect(filteredTapeNodes[0]).toEqual(tape[0]);
            expect(filteredTapeNodes[1]).toEqual(tape[1]);
            expect(filteredTapeNodes[2]).toEqual({
                id: 3,
                type: 'kernel',
                name: 'node3',
                inputAndArgs: {
                    inputs: { intermediate0: intermediate0, intermediate1: intermediate1, x1: x1 },
                },
                output: { y: y },
                gradient: null
            });
        });
    };
    test_util.describeMathCPU('tape_util.getFilteredNodesXToY', [tests]);
}
{
    var tests = function (it) {
        it('Throws if gradient is not defined', function (math) {
            var x = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(1);
            var dy = ndarray_1.Scalar.new(1);
            var accumulatedGradientsMap = {};
            accumulatedGradientsMap[y.id] = dy;
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: null
                }];
            expect(function () { return tape_util.backpropagateGradients(accumulatedGradientsMap, tape); })
                .toThrowError();
        });
        it('basic backprop with 1 node', function (math) {
            var x = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(1);
            var dy = ndarray_1.Scalar.new(1);
            var accumulatedGradientsMap = {};
            accumulatedGradientsMap[y.id] = dy;
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: function (dy, y) {
                        return { x: function () { return math.add(dy, ndarray_1.Scalar.new(1)); } };
                    }
                }];
            tape_util.backpropagateGradients(accumulatedGradientsMap, tape);
            test_util.expectArraysClose(accumulatedGradientsMap[x.id], [2]);
        });
        it('basic backprop with 2 nodes', function (math) {
            var x = ndarray_1.Scalar.new(0);
            var intermediate = ndarray_1.Scalar.new(1);
            var y = ndarray_1.Scalar.new(2);
            var dy = ndarray_1.Scalar.new(1);
            var accumulatedGradientsMap = {};
            accumulatedGradientsMap[y.id] = dy;
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: intermediate,
                    gradient: function (dy, y) {
                        return { x: function () { return math.add(dy, ndarray_1.Scalar.new(1)); } };
                    }
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate: intermediate },
                    },
                    output: y,
                    gradient: function (dy, y) {
                        return { intermediate: function () { return math.add(dy, ndarray_1.Scalar.new(1)); } };
                    }
                }
            ];
            tape_util.backpropagateGradients(accumulatedGradientsMap, tape);
            test_util.expectArraysClose(accumulatedGradientsMap[x.id], [3]);
        });
        it('basic backprop with a split node accumulates gradients', function (math) {
            var x = ndarray_1.Scalar.new(0);
            var intermediate1 = ndarray_1.Scalar.new(1);
            var intermediate2 = ndarray_1.Scalar.new(2);
            var y = ndarray_1.Scalar.new(3);
            var dy = ndarray_1.Scalar.new(1);
            var accumulatedGradientsMap = {};
            accumulatedGradientsMap[y.id] = dy;
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: intermediate1,
                    gradient: function (dy, y) {
                        return { x: function () { return math.add(dy, ndarray_1.Scalar.new(1)); } };
                    }
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: intermediate2,
                    gradient: function (dy, y) {
                        return { x: function () { return math.add(dy, ndarray_1.Scalar.new(1)); } };
                    }
                },
                {
                    id: 2,
                    type: 'kernel',
                    name: 'node2',
                    inputAndArgs: {
                        inputs: { intermediate1: intermediate1, intermediate2: intermediate2 },
                    },
                    output: y,
                    gradient: function (dy, y) {
                        return {
                            intermediate1: function () { return math.add(dy, ndarray_1.Scalar.new(1)); },
                            intermediate2: function () { return math.add(dy, ndarray_1.Scalar.new(1)); }
                        };
                    }
                }
            ];
            tape_util.backpropagateGradients(accumulatedGradientsMap, tape);
            test_util.expectArraysClose(accumulatedGradientsMap[x.id], [dy.dataSync()[0] + 5]);
        });
        it('basic backprop with a multi-output split node accumulates gradients', function (math) {
            var x = ndarray_1.Scalar.new(0);
            var intermediate1 = ndarray_1.Scalar.new(1);
            var intermediate2 = ndarray_1.Scalar.new(2);
            var y = ndarray_1.Scalar.new(3);
            var dy = ndarray_1.Scalar.new(1);
            var accumulatedGradientsMap = {};
            accumulatedGradientsMap[y.id] = dy;
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: { intermediate1: intermediate1, intermediate2: intermediate2 },
                    gradient: function (dy, y) {
                        return {
                            x: function () {
                                return math.multiply(dy['intermediate1'], dy['intermediate2']);
                            }
                        };
                    }
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate1: intermediate1, intermediate2: intermediate2 },
                    },
                    output: y,
                    gradient: function (dy, y) {
                        return {
                            intermediate1: function () { return math.add(dy, ndarray_1.Scalar.new(2)); },
                            intermediate2: function () { return math.add(dy, ndarray_1.Scalar.new(3)); }
                        };
                    }
                }
            ];
            tape_util.backpropagateGradients(accumulatedGradientsMap, tape);
            test_util.expectArraysClose(accumulatedGradientsMap[x.id], [(dy.get() + 2) * (dy.get() + 3)]);
        });
    };
    test_util.describeMathCPU('tape_util.backpropagateGradients', [tests]);
}
{
    var tests = function (it) {
        it('no inputs', function (math) {
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: {},
                    },
                    output: y,
                    gradient: null
                }];
            var varList = [];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([]);
        });
        it('no variable inputs', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: null
                }];
            var varList = [];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([]);
        });
        it('one variable input, not in varList', function (math) {
            var x = ndarray_1.variable(ndarray_1.Scalar.new(1));
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: null
                }];
            var varList = [];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([]);
        });
        it('one variable input in varList', function (math) {
            var x = ndarray_1.variable(ndarray_1.Scalar.new(1));
            var y = ndarray_1.Scalar.new(2);
            var tape = [{
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x: x },
                    },
                    output: y,
                    gradient: null
                }];
            var varList = [x];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([x]);
        });
        it('multiple inputs from multiple ops, no variables', function (math) {
            var x1 = ndarray_1.Scalar.new(1);
            var intermediate1 = ndarray_1.Scalar.new(0);
            var x2 = ndarray_1.Scalar.new(0);
            var y = ndarray_1.Scalar.new(2);
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x1: x1 },
                    },
                    output: intermediate1,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate1: intermediate1, x2: x2 },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var varList = [];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([]);
        });
        it('multiple inputs from multiple ops, two variables', function (math) {
            var x1 = ndarray_1.variable(ndarray_1.Scalar.new(1));
            var intermediate1 = ndarray_1.Scalar.new(0);
            var x2 = ndarray_1.variable(ndarray_1.Scalar.new(0));
            var y = ndarray_1.Scalar.new(2);
            var notInTape = ndarray_1.variable(ndarray_1.Scalar.new(3));
            expect(notInTape).not.toBeNull();
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x1: x1 },
                    },
                    output: intermediate1,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate1: intermediate1, x2: x2 },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var varList = [x1, x2];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([x1, x2]);
        });
        it('multiple inputs, two variables, only one in varList', function (math) {
            var x1 = ndarray_1.variable(ndarray_1.Scalar.new(1));
            var intermediate1 = ndarray_1.Scalar.new(0);
            var x2 = ndarray_1.variable(ndarray_1.Scalar.new(0));
            var y = ndarray_1.Scalar.new(2);
            var notInTape = ndarray_1.variable(ndarray_1.Scalar.new(3));
            expect(notInTape).not.toBeNull();
            var tape = [
                {
                    id: 0,
                    type: 'kernel',
                    name: 'node0',
                    inputAndArgs: {
                        inputs: { x1: x1 },
                    },
                    output: intermediate1,
                    gradient: null
                },
                {
                    id: 1,
                    type: 'kernel',
                    name: 'node1',
                    inputAndArgs: {
                        inputs: { intermediate1: intermediate1, x2: x2 },
                    },
                    output: y,
                    gradient: null
                }
            ];
            var varList = [x2];
            var inputs = tape_util.computeVariableInputs(tape, varList);
            expect(inputs).toEqual([x2]);
        });
    };
    test_util.describeMathCPU('tape_util.computeInputs', [tests]);
}
{
    var tests = function (it) {
        it('null input returns empty array', function (math) {
            var results = tape_util.extractNDArraysFromScopeResult(null);
            expect(results).toEqual([]);
        });
        it('ndarray input returns one element array', function (math) {
            var x = ndarray_1.Scalar.new(1);
            var results = tape_util.extractNDArraysFromScopeResult(x);
            expect(results).toEqual([x]);
        });
        it('name array map returns flattened array', function (math) {
            var x1 = ndarray_1.Scalar.new(1);
            var x2 = ndarray_1.Scalar.new(3);
            var x3 = ndarray_1.Scalar.new(4);
            var results = tape_util.extractNDArraysFromScopeResult({ x1: x1, x2: x2, x3: x3 });
            expect(results).toEqual([x1, x2, x3]);
        });
    };
    test_util.describeMathCPU('tape_util.extractNDArraysFromScopeResult', [tests]);
}
{
    var tests_1 = function (it) {
        it('pass through when all inputs are defined', function () {
            var x1 = ndarray_1.Scalar.new(1);
            var x2 = ndarray_1.Scalar.new(2);
            var config = {
                inputs: { x1: x1, x2: x2 },
            };
            expect(tape_util.stripUndefinedInputsFromInputConfig(config)).toEqual({
                inputs: { x1: x1, x2: x2 }
            });
        });
        it('strips undefined inputs', function () {
            var x1 = ndarray_1.Scalar.new(1);
            var x4 = ndarray_1.Scalar.new(2);
            var config = {
                inputs: { x1: x1, x2: undefined, x3: undefined, x4: x4 },
            };
            expect(tape_util.stripUndefinedInputsFromInputConfig(config)).toEqual({
                inputs: { x1: x1, x4: x4 }
            });
        });
        test_util.describeMathCPU('tape_util.extractNDArraysFromScopeResult', [tests_1]);
    };
}
//# sourceMappingURL=tape_util_test.js.map