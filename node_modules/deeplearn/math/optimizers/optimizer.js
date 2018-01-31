"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var session_util = require("../../graph/session_util");
var tensor_array_map_1 = require("../../graph/tensor_array_map");
var ndarray_1 = require("../../math/ndarray");
var Optimizer = (function () {
    function Optimizer(learningRate, specifiedVariableList) {
        this.learningRate = learningRate;
        this.variableGradients = new tensor_array_map_1.TensorArrayMap();
        if (specifiedVariableList != null) {
            this.specifiedVariableNodes = specifiedVariableList;
        }
        this.one = environment_1.ENV.math.keep(ndarray_1.Scalar.new(1));
    }
    Optimizer.prototype.minimize = function (f, returnCost, varList) {
        if (returnCost === void 0) { returnCost = false; }
        var _a = this.computeGradients(f, varList), value = _a.value, gradients = _a.gradients;
        this.applyGradients(gradients);
        var varNames = Object.keys(gradients);
        varNames.forEach(function (varName) { return gradients[varName].dispose(); });
        if (returnCost) {
            return value;
        }
        else {
            value.dispose();
            return null;
        }
    };
    Optimizer.prototype.computeGradients = function (f, varList) {
        return environment_1.ENV.math.variableGradients(f, varList);
    };
    Optimizer.prototype.beforeBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        this.variableNodes = this.specifiedVariableNodes == null ?
            session_util.getVariableNodesFromEvaluationSet(runtime.nodes) :
            this.specifiedVariableNodes;
        if (batchSize !== this.prevBatchSize) {
            if (this.cGraph != null) {
                this.cGraph.dispose();
            }
            this.prevBatchSize = batchSize;
            this.cGraph = math.keep(ndarray_1.Scalar.new(-this.learningRate / batchSize));
        }
        this.variableNodes.forEach(function (node) { return _this.variableGradients.set(node.output, math.keep(ndarray_1.NDArray.zeros(node.output.shape))); });
    };
    Optimizer.prototype.afterExample = function (math, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        math.scope(function (keep) {
            _this.variableNodes.forEach(function (node) {
                var gradient = gradientArrayMap.get(node.output);
                var accumulatedGradient = _this.variableGradients.get(node.output);
                _this.variableGradients.set(node.output, keep(math.add(gradient, accumulatedGradient)));
                accumulatedGradient.dispose();
            });
        });
    };
    Optimizer.prototype.dispose = function () {
        if (this.cGraph != null) {
            this.cGraph.dispose();
        }
        this.one.dispose();
        if (this.variableNodes != null) {
            this.variableNodes.forEach(function (node) {
                node.data.dispose();
            });
        }
        if (this.specifiedVariableNodes != null) {
            this.specifiedVariableNodes.forEach(function (node) {
                node.data.dispose();
            });
        }
    };
    return Optimizer;
}());
exports.Optimizer = Optimizer;
//# sourceMappingURL=optimizer.js.map