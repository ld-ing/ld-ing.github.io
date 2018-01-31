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
var decorators_1 = require("./decorators");
var Ops = (function () {
    function Ops() {
    }
    Ops.batchNormalization2D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 2, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 2 || mean.rank === 1, "Error in batchNormalization2D: mean must be rank 2 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 2 || variance.rank === 1, "Error in batchNormalization2D: variance must be rank 2 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 2 || scale.rank === 1, "Error in batchNormalization2D: scale must be rank 2 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 2 || offset.rank === 1, "Error in batchNormalization2D: offset must be rank 2 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return environment_1.ENV.engine.executeKernel('BatchNorm2D', {
            inputs: { x: x, mean: mean, variance: variance, scale: scale, offset: offset },
            args: { varianceEpsilon: varianceEpsilon }
        });
    };
    Ops.batchNormalization3D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 3, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 3 || mean.rank === 1, "Error in batchNormalization3D: mean must be rank 3 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 3 || variance.rank === 1, "Error in batchNormalization3D: variance must be rank 3 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 3 || scale.rank === 1, "Error in batchNormalization3D: scale must be rank 3 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 3 || offset.rank === 1, "Error in batchNormalization3D: offset must be rank 3 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return environment_1.ENV.engine.executeKernel('BatchNorm3D', {
            inputs: { x: x, mean: mean, variance: variance, scale: scale, offset: offset },
            args: { varianceEpsilon: varianceEpsilon }
        });
    };
    Ops.batchNormalization4D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 4, "Error in batchNormalization4D: x must be rank 4 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 4 || mean.rank === 1, "Error in batchNormalization4D: mean must be rank 4 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 4 || variance.rank === 1, "Error in batchNormalization4D: variance must be rank 4 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 4 || scale.rank === 1, "Error in batchNormalization4D: scale must be rank 4 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 4 || offset.rank === 1, "Error in batchNormalization4D: offset must be rank 4 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return environment_1.ENV.engine.executeKernel('BatchNorm4D', {
            inputs: { x: x, mean: mean, variance: variance, scale: scale, offset: offset },
            args: { varianceEpsilon: varianceEpsilon }
        });
    };
    Ops.batchNormalization = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        if (x.rank === 0) {
            throw new Error("Batchnorm for scalar is not supported");
        }
        else if (x.rank === 1) {
            throw new Error("Batchnorm for rank 1 is not yet implemented");
        }
        else if (x.rank === 2) {
            return Ops.batchNormalization2D(x, mean, variance, varianceEpsilon, scale, offset);
        }
        else if (x.rank === 3) {
            return Ops.batchNormalization3D(x, mean, variance, varianceEpsilon, scale, offset);
        }
        else if (x.rank === 4) {
            return Ops.batchNormalization4D(x, mean, variance, varianceEpsilon, scale, offset);
        }
        else {
            throw new Error("Batchnorm for rank " + x.rank + " is not yet implemented");
        }
    };
    __decorate([
        decorators_1.operation
    ], Ops, "batchNormalization2D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "batchNormalization3D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "batchNormalization4D", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=batchnorm.js.map