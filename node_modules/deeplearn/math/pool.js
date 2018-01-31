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
var conv_util = require("./conv_util");
var decorators_1 = require("./decorators");
var Ops = (function () {
    function Ops() {
    }
    Ops.maxPool = function (x, filterSize, strides, pad, dimRoundingMode) {
        var x4D = x;
        var reshapedTo4D = false;
        if (x.rank === 3) {
            reshapedTo4D = true;
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in maxPool: input must be rank 4 but got rank " + x4D.rank + ".");
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in maxPool: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(x4D.shape, filterSize, strides, pad, dimRoundingMode);
        var gradients = function (dy, y) {
            return { x: function () { return Ops.maxPoolBackprop(dy, x4D, filterSize, strides, pad); } };
        };
        var res = environment_1.ENV.engine.executeKernel('MaxPool', { inputs: { x: x4D }, args: { convInfo: convInfo } }, gradients);
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    Ops.maxPoolBackprop = function (dy, input, filterSize, strides, pad, dimRoundingMode) {
        util.assert(input.rank === dy.rank, "Rank of input (" + input.rank + ") does not match rank of dy (" + dy.rank + ")");
        var input4D = input;
        var dy4D = dy;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        util.assert(dy4D.rank === 4, "Error in maxPoolBackprop: dy must be rank 4 but got rank " +
            (dy4D.rank + "."));
        util.assert(input4D.rank === 4, "Error in maxPoolBackprop: input must be rank 4 but got rank " +
            (input4D.rank + "."));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in maxPoolBackprop: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad, dimRoundingMode);
        var res = environment_1.ENV.engine.executeKernel('MaxPoolBackprop', { inputs: { dy: dy4D, x: input4D }, args: { convInfo: convInfo } });
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    Ops.minPool = function (input, filterSize, strides, pad, dimRoundingMode) {
        var input4D = input;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
        }
        util.assert(input4D.rank === 4, "Error in minPool: x must be rank 4 but got rank " + input4D.rank + ".");
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in minPool: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad, dimRoundingMode);
        var res = environment_1.ENV.engine.executeKernel('MinPool', { inputs: { x: input4D }, args: { convInfo: convInfo } });
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    Ops.avgPool = function (x, filterSize, strides, pad, dimRoundingMode) {
        var x4D = x;
        var reshapedTo4D = false;
        if (x.rank === 3) {
            reshapedTo4D = true;
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in avgPool: x must be rank 4 but got rank " + x4D.rank + ".");
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in avgPool: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computePool2DInfo(x4D.shape, filterSize, strides, pad);
        var gradients = function (dy, y) {
            return { x: function () { return Ops.avgPoolBackprop(dy, x4D, filterSize, strides, pad); } };
        };
        var res = environment_1.ENV.engine.executeKernel('AvgPool', { inputs: { x: x4D }, args: { convInfo: convInfo } }, gradients);
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    Ops.avgPoolBackprop = function (dy, input, filterSize, strides, pad) {
        util.assert(input.rank === dy.rank, "Rank of input (" + input.rank + ") does not match rank of dy (" + dy.rank + ")");
        var input4D = input;
        var dy4D = dy;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        util.assert(dy4D.rank === 4, "Error in avgPoolBackprop: dy must be rank 4 but got rank " +
            (dy4D.rank + "."));
        util.assert(input4D.rank === 4, "Error in avgPoolBackprop: input must be rank 4 but got rank " +
            (input4D.rank + "."));
        var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
        var res = environment_1.ENV.engine.executeKernel('AvgPoolBackprop', { inputs: { dy: dy4D, x: input4D }, args: { convInfo: convInfo } });
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    __decorate([
        decorators_1.operation
    ], Ops, "maxPool", null);
    __decorate([
        decorators_1.operation
    ], Ops, "maxPoolBackprop", null);
    __decorate([
        decorators_1.operation
    ], Ops, "minPool", null);
    __decorate([
        decorators_1.operation
    ], Ops, "avgPool", null);
    __decorate([
        decorators_1.operation
    ], Ops, "avgPoolBackprop", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=pool.js.map