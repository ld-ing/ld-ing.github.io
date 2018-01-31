"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var ndarray_1 = require("../ndarray");
var kernel_registry = require("./kernel_registry");
var tape_util = require("./tape_util");
var BackendEngine = (function () {
    function BackendEngine(backend, safeMode) {
        this.backend = backend;
        this.safeMode = safeMode;
        this.nextTapeNodeId = 0;
        this.gradientScopeCount = 0;
        this.customGradientDepth = 0;
        this.debugMode = false;
        this.activeScope = { keep: [], track: [] };
        this.scopeStack = [this.activeScope];
    }
    BackendEngine.prototype.enableDebugMode = function () {
        this.debugMode = true;
    };
    BackendEngine.prototype.executeKernel = function (kernelName, config, grad) {
        var start;
        if (this.debugMode) {
            start = performance.now();
        }
        var result = kernel_registry.executeKernel(this.backend, kernelName, config);
        if (this.debugMode) {
            var vals = result.dataSync();
            var time = util.rightPad(performance.now() - start + "ms", 9);
            var paddedName = util.rightPad(kernelName, 25);
            var rank = result.rank;
            var size = result.size;
            var shape = util.rightPad(result.shape.toString(), 14);
            console.log("%c" + paddedName + "\t%c" + time + "\t%c" + rank + "D " + shape + "\t%c" + size, 'font-weight:bold', 'color:red', 'color:blue', 'color: orange');
            util.checkForNaN(vals, result.dtype, name);
        }
        if (this.activeTape != null && this.customGradientDepth === 0) {
            config = tape_util.stripUndefinedInputsFromInputConfig(config);
            var evaluatedNode = {
                id: this.nextTapeNodeId++,
                type: 'kernel',
                name: "kernel: " + kernelName,
                kernel: kernelName,
                inputAndArgs: config,
                output: result,
                gradient: grad
            };
            this.activeTape.push(evaluatedNode);
        }
        return result;
    };
    BackendEngine.prototype.customGradient = function (f, inputs, name) {
        this.customGradientDepth++;
        var gradientsFunc;
        var gradientsMode = true;
        var result = this.scope('customGradient', function () {
            var _a = f(), value = _a.value, gradients = _a.gradients;
            gradientsFunc = gradients;
            return value;
        }, gradientsMode);
        this.customGradientDepth--;
        if (this.activeTape != null && this.customGradientDepth === 0) {
            var evaluatedNode = {
                id: this.nextTapeNodeId++,
                type: 'customGradient',
                name: name,
                inputAndArgs: { inputs: inputs },
                output: result,
                gradient: gradientsFunc
            };
            this.activeTape.push(evaluatedNode);
        }
        return result;
    };
    BackendEngine.prototype.gradients = function (f, xs, returnValue) {
        var _this = this;
        var gradientsMode = true;
        var result = this.scope('gradients', function () {
            var y = f();
            if (y.rank !== 0) {
                throw new Error("Cannot compute gradient of non-scalar y output of f(). " +
                    ("Got y with rank " + y.rank + " and shape " + y.shape + "."));
            }
            var gradients = _this.gradientWrt(y, xs);
            if (returnValue) {
                return [y].concat(gradients);
            }
            else {
                return gradients;
            }
        }, gradientsMode);
        if (returnValue) {
            return { value: result[0], gradients: result.slice(1) };
        }
        else {
            return result;
        }
    };
    BackendEngine.prototype.vjp = function (f, xs, dy) {
        var _this = this;
        var gradientsMode = true;
        return this.scope('vjp', function () {
            var y = f();
            if (!util.arraysEqual(y.shape, dy.shape)) {
                throw new Error("Cannot compute vector jacobian product, " +
                    ("y shape (" + y.shape + ") does not match dy shape (" + dy.shape + ")."));
            }
            return _this.gradientWrt(y, xs, dy);
        }, gradientsMode);
    };
    BackendEngine.prototype.variableGradientsAndValue = function (f, varList) {
        var _this = this;
        var gradientsMode = true;
        var variableNames;
        var result = this.scope('gradients', function () {
            var y = f();
            if (y.rank !== 0) {
                throw new Error("Cannot compute gradient of non-scalar y output of f(). " +
                    ("Got y with rank " + y.rank + " and shape " + y.shape + "."));
            }
            var inputVariables = tape_util.computeVariableInputs(_this.activeTape, varList);
            variableNames = inputVariables.map(function (variable) { return variable.name; });
            var gradients = inputVariables.length === 0 ?
                [] :
                _this.gradientWrt(y, inputVariables);
            return [y].concat(gradients);
        }, gradientsMode);
        var gradients = {};
        for (var i = 0; i < variableNames.length; i++) {
            gradients[variableNames[i]] = result[i + 1];
        }
        return { value: result[0], gradients: gradients };
    };
    BackendEngine.prototype.gradientWrt = function (y, xs, dy) {
        var filteredTape = tape_util.getFilteredNodesXToY(this.activeTape, xs, y);
        if (filteredTape.length === 0) {
            throw new Error("Cannot compute gradient: y is not a function of xs." +
                "Make sure the xs you are computing gradients with respect " +
                "to are used inside the gradient function.");
        }
        var arrayAccumulatedGradientMap = {};
        arrayAccumulatedGradientMap[y.id] =
            dy == null ? ndarray_1.Scalar.new(1, 'float32') : dy;
        tape_util.backpropagateGradients(arrayAccumulatedGradientMap, filteredTape);
        var gradients = xs.map(function (x) { return arrayAccumulatedGradientMap[x.id]; });
        gradients.forEach(function (grad, i) {
            if (grad == null) {
                throw new Error("Gradient error: y was not a function of xs[" + i + "]");
            }
        });
        return gradients;
    };
    BackendEngine.prototype.scope = function (name, scopeFn, gradientsMode) {
        var _this = this;
        this.startScope(gradientsMode);
        var keepFn = function (ndarray) { return _this.keep(ndarray); };
        var trackFn = function (ndarray) { return ndarray; };
        var result = scopeFn(keepFn, trackFn);
        if (result instanceof Promise) {
            result.then(function (r) { return _this.endScope(r, gradientsMode); });
            return result;
        }
        else {
            this.endScope(result, gradientsMode);
            return result;
        }
    };
    BackendEngine.prototype.startScope = function (gradientsMode) {
        if (gradientsMode && this.gradientScopeCount === 0) {
            this.activeTape = [];
        }
        if (gradientsMode) {
            this.gradientScopeCount++;
        }
        var newScopeArrays = { keep: [], track: [] };
        this.scopeStack.push(newScopeArrays);
        this.activeScope = newScopeArrays;
    };
    BackendEngine.prototype.endScope = function (result, gradientsMode) {
        var _this = this;
        if (gradientsMode) {
            this.gradientScopeCount--;
            if (this.gradientScopeCount === 0) {
                this.activeTape = null;
            }
        }
        var arraysToKeep = this.activeScope.keep;
        var arraysToTrackInParent = tape_util.extractNDArraysFromScopeResult(result);
        arraysToKeep = arraysToKeep.concat(arraysToTrackInParent);
        for (var i = 0; i < this.activeScope.track.length; i++) {
            var ndarray = this.activeScope.track[i];
            if (util.isNDArrayInList(ndarray, arraysToKeep)) {
                continue;
            }
            if (this.activeTape != null) {
                arraysToTrackInParent.push(ndarray);
            }
            else {
                ndarray.dispose();
            }
        }
        this.scopeStack.pop();
        this.activeScope = this.scopeStack.length === 0 ?
            null :
            this.scopeStack[this.scopeStack.length - 1];
        arraysToTrackInParent.forEach(function (ndarray) {
            if (!util.isNDArrayInList(ndarray, _this.activeScope.keep)) {
                _this.track(ndarray);
            }
        });
    };
    BackendEngine.prototype.keep = function (result) {
        if (this.scopeStack.length === 1) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
        }
        this.activeScope.keep.push(result);
        return result;
    };
    BackendEngine.prototype.track = function (result) {
        if (this.scopeStack.length === 1) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
        }
        this.activeScope.track.push(result);
        return result;
    };
    BackendEngine.prototype.getBackend = function () {
        return this.backend;
    };
    return BackendEngine;
}());
exports.BackendEngine = BackendEngine;
//# sourceMappingURL=backend_engine.js.map