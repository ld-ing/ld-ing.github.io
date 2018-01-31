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
    Ops.logSumExp = function (input, axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, input.shape);
        var xMax = input.max(axes, true);
        var a = input.sub(xMax);
        var b = a.exp();
        var c = b.sum(axes);
        var d = c.log();
        var res = xMax.reshape(d.shape).add(d);
        if (keepDims) {
            var newShape = axis_util.expandShapeToKeepDim(res.shape, axes);
            return res.reshape(newShape);
        }
        return res;
    };
    Ops.sum = function (x, axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        return environment_1.ENV.math.customGradient('sum', function () {
            var permutation = axis_util.getAxesPermutation(axes, x.rank);
            var reductionAxes = axes;
            var permutedX = x;
            if (permutation != null) {
                permutedX = x.transpose(permutation);
                reductionAxes =
                    axis_util.getInnerMostAxes(reductionAxes.length, x.rank);
            }
            var value = environment_1.ENV.engine.executeKernel('Sum', { inputs: { x: permutedX }, args: { axes: reductionAxes } });
            if (keepDims) {
                var newShape = axis_util.expandShapeToKeepDim(value.shape, axes);
                value = value.reshape(newShape);
            }
            var gradients = function (dy) {
                var expandedDyShape = x.shape.slice();
                axes.forEach(function (axis) {
                    expandedDyShape[axis] = 1;
                });
                var expandedDy = dy.reshape(expandedDyShape);
                var derX = function () { return expandedDy.mul(ndarray_1.NDArray.ones(x.shape, 'float32')); };
                return { x: derX };
            };
            return { value: value, gradients: gradients };
        }, { x: x });
    };
    Ops.mean = function (x, axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var shapes = axis_util.computeOutAndReduceShapes(x.shape, axes);
        var reduceShape = shapes[1];
        var reduceSize = util.sizeFromShape(reduceShape);
        return environment_1.ENV.math.customGradient('mean', function () {
            var reduceSizeScalar = ndarray_1.Scalar.new(reduceSize);
            var res = x.div(reduceSizeScalar);
            var value = res.sum(axis, keepDims);
            var gradients = function (dy) {
                var expandedDyShape = x.shape.slice();
                axes.forEach(function (axis) {
                    expandedDyShape[axis] = 1;
                });
                var expandedDy = dy.reshape(expandedDyShape);
                var derX = function () { return expandedDy.mul(ndarray_1.NDArray.ones(x.shape, 'float32'))
                    .div(reduceSizeScalar); };
                return { x: derX };
            };
            return { value: value, gradients: gradients };
        }, { x: x });
    };
    Ops.min = function (x, axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var origAxes = axis_util.parseAxisParam(axis, x.shape);
        var axes = origAxes;
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        if (permutedAxes != null) {
            x = x.transpose(permutedAxes);
            axes = axis_util.getInnerMostAxes(axes.length, x.rank);
        }
        var res = environment_1.ENV.engine.executeKernel('Min', { inputs: { x: x }, args: { axes: axes } });
        if (keepDims) {
            var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
            return res.reshape(newShape);
        }
        return res;
    };
    Ops.max = function (x, axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var origAxes = axis_util.parseAxisParam(axis, x.shape);
        var axes = origAxes;
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        if (permutedAxes != null) {
            x = x.transpose(permutedAxes);
            axes = axis_util.getInnerMostAxes(axes.length, x.rank);
        }
        var res = environment_1.ENV.engine.executeKernel('Max', { inputs: { x: x }, args: { axes: axes } });
        if (keepDims) {
            var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
            return res.reshape(newShape);
        }
        return res;
    };
    Ops.argMin = function (x, axis) {
        if (axis === void 0) { axis = null; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        if (permutedAxes != null) {
            x = x.transpose(permutedAxes);
            axes = axis_util.getInnerMostAxes(axes.length, x.rank);
        }
        return environment_1.ENV.engine.executeKernel('ArgMin', { inputs: { x: x }, args: { axes: axes } });
    };
    Ops.argMax = function (x, axis) {
        if (axis === void 0) { axis = null; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
        if (permutedAxes != null) {
            x = x.transpose(permutedAxes);
            axes = axis_util.getInnerMostAxes(axes.length, x.rank);
        }
        return environment_1.ENV.engine.executeKernel('ArgMax', { inputs: { x: x }, args: { axes: axes } });
    };
    Ops.argMaxEquals = function (x1, x2) {
        util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
        return x1.argMax().equal(x2.argMax());
    };
    Ops.moments = function (x, axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        var axes = axis_util.parseAxisParam(axis, x.shape);
        var mean = x.mean(axes, keepDims);
        var keepDimsShape = mean.shape;
        if (!keepDims) {
            keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
        }
        var devSquared = x.toFloat().sub(mean.reshape(keepDimsShape)).square();
        var variance = devSquared.mean(axes, keepDims);
        return { mean: mean, variance: variance };
    };
    __decorate([
        decorators_1.operation
    ], Ops, "logSumExp", null);
    __decorate([
        decorators_1.operation
    ], Ops, "sum", null);
    __decorate([
        decorators_1.operation
    ], Ops, "mean", null);
    __decorate([
        decorators_1.operation
    ], Ops, "min", null);
    __decorate([
        decorators_1.operation
    ], Ops, "max", null);
    __decorate([
        decorators_1.operation
    ], Ops, "argMin", null);
    __decorate([
        decorators_1.operation
    ], Ops, "argMax", null);
    __decorate([
        decorators_1.operation
    ], Ops, "argMaxEquals", null);
    __decorate([
        decorators_1.operation
    ], Ops, "moments", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=reduction_ops.js.map