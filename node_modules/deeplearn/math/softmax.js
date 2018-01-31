"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var util = require("../util");
var axis_util = require("./axis_util");
var decorators_1 = require("./decorators");
var ndarray_1 = require("./ndarray");
var Ops = (function () {
    function Ops() {
    }
    Ops.softmax = function (logits, dim) {
        if (dim === void 0) { dim = -1; }
        if (dim === -1) {
            dim = logits.rank - 1;
        }
        if (dim !== logits.rank - 1) {
            throw Error('Softmax along a non-last dimension is not yet supported. ' +
                ("Logits was rank " + logits.rank + " and dim was " + dim));
        }
        var gradients = function (dy, y) {
            return {
                logits: function () {
                    var dyTimesY = dy.mul(y);
                    var keepDims = true;
                    return dyTimesY.sub(dyTimesY.sum([dim], keepDims).mul(y));
                }
            };
        };
        return environment_1.ENV.math.customGradient('softmax', function () {
            var keepDims = true;
            var lse = logits.logSumExp([dim], keepDims);
            var logResult = logits.toFloat().sub(lse);
            var value = logResult.exp();
            return { value: value, gradients: gradients };
        }, { logits: logits });
    };
    Ops.softmaxCrossEntropy = function (labels, logits, dim) {
        if (dim === void 0) { dim = -1; }
        util.assertShapesMatch(labels.shape, logits.shape, 'Error in softmaxCrossEntropy: ');
        if (dim === -1) {
            dim = logits.rank - 1;
        }
        if (dim !== logits.rank - 1) {
            throw Error("Softmax cross entropy along a non-last dimension is not yet " +
                ("supported. Labels / logits was rank " + logits.rank + " ") +
                ("and dim was " + dim));
        }
        return environment_1.ENV.math.customGradient('softmaxCrossEntropy', function () {
            var softmaxLogits = logits.softmax(dim);
            var costVector = ndarray_1.Scalar.new(1e-5).add(softmaxLogits).log().mul(labels).neg();
            var value = costVector.sum([dim]);
            var gradients = function (dy, y) {
                var dyShape = axis_util.expandShapeToKeepDim(dy.shape, [dim]);
                return {
                    logits: function () {
                        return dy.reshape(dyShape).mul(softmaxLogits.sub(labels.toFloat()));
                    },
                    labels: function () { return dy.reshape(dyShape).mul(labels.sub(softmaxLogits)); }
                };
            };
            return { value: value, gradients: gradients };
        }, { labels: labels, logits: logits });
    };
    __decorate([
        decorators_1.operation
    ], Ops, "softmax", null);
    __decorate([
        decorators_1.operation
    ], Ops, "softmaxCrossEntropy", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=softmax.js.map