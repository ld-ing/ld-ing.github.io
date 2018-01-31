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
    Ops.conv1d = function (input, filter, bias, stride, pad, dimRoundingMode) {
        var input3D = input;
        var reshapedTo3D = false;
        if (input.rank === 2) {
            reshapedTo3D = true;
            input3D = input.as3D(1, input.shape[0], input.shape[1]);
        }
        util.assert(input3D.rank === 3, "Error in conv1d: input must be rank 3, but got rank " + input3D.rank + ".");
        util.assert(filter.rank === 3, "Error in conv1d: filter must be rank 3, but got rank " +
            (filter.rank + "."));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv1d: bias must be rank 1, but got rank " +
                (bias.rank + "."));
        }
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv1d: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        util.assert(input3D.shape[2] === filter.shape[1], "Error in conv1d: depth of input (" + input3D.shape[2] + ") must match  " +
            ("input depth for filter " + filter.shape[1] + "."));
        var filter4D = filter.as4D(1, filter.shape[0], filter.shape[1], filter.shape[2]);
        var input4D = input3D.as4D(input3D.shape[0], 1, input3D.shape[1], input3D.shape[2]);
        var strides = [1, stride];
        var res = Ops.conv2d(input4D, filter4D, bias, strides, pad, dimRoundingMode);
        if (reshapedTo3D) {
            return res.as2D(res.shape[2], res.shape[3]);
        }
        return res.as3D(res.shape[0], res.shape[2], res.shape[3]);
    };
    Ops.conv2d = function (x, filter, bias, strides, pad, dimRoundingMode) {
        var x4D = x;
        var reshapedTo4D = false;
        if (x.rank === 3) {
            reshapedTo4D = true;
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in conv2d: input must be rank 4, but got rank " + x4D.rank + ".");
        util.assert(filter.rank === 4, "Error in conv2d: filter must be rank 4, but got rank " +
            (filter.rank + "."));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv2d: bias must be rank 1, but got rank " +
                (bias.rank + "."));
        }
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv2d: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        util.assert(x4D.shape[3] === filter.shape[2], "Error in conv2d: depth of input (" + x4D.shape[3] + ") must match  " +
            ("input depth for filter " + filter.shape[2] + "."));
        var convInfo = conv_util.computeConv2DInfo(x4D.shape, filter.shape, strides, pad, dimRoundingMode);
        var gradients = function (dy, y) {
            return {
                x: function () { return Ops.conv2dDerInput(x4D.shape, dy, filter, strides, pad); },
                filter: function () { return Ops.conv2dDerFilter(x4D, dy, filter.shape, strides, pad); },
                bias: function () { return Ops.conv2dDerBias(dy); }
            };
        };
        var res = environment_1.ENV.engine.executeKernel('Conv2D', { inputs: { x: x4D, filter: filter, bias: bias }, args: { convInfo: convInfo } }, gradients);
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    Ops.conv2dDerInput = function (xShape, dy, filter, strides, pad, dimRoundingMode) {
        util.assert(xShape.length === dy.rank, "Length of inShape " +
            ("(" + xShape.length + ") and rank of dy (" + dy.rank + ") must match"));
        var xShape4D = xShape;
        var dy4D = dy;
        var reshapedTo4D = false;
        if (dy.rank === 3) {
            reshapedTo4D = true;
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
            xShape4D = [1, xShape[0], xShape[1], xShape[2]];
        }
        var inDepth = xShape4D[3];
        var outDepth = dy4D.shape[3];
        util.assert(xShape4D.length === 4, "Error in conv2dDerInput: inShape must be length 4, but got length " +
            (xShape4D.length + "."));
        util.assert(dy4D.rank === 4, "Error in conv2dDerInput: dy must be rank 4, but got " +
            ("rank " + dy4D.rank));
        util.assert(filter.rank === 4, "Error in conv2dDerInput: filter must be rank 4, but got " +
            ("rank " + filter.rank));
        util.assert(inDepth === filter.shape[2], "Error in conv2dDerInput: depth of input (" + inDepth + ") must " +
            ("match input depth for filter " + filter.shape[2] + "."));
        util.assert(outDepth === filter.shape[3], "Error in conv2dDerInput: depth of output (" + outDepth + ") must" +
            ("match output depth for filter " + filter.shape[3] + "."));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv2dDerInput: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computeConv2DInfo(xShape4D, filter.shape, strides, pad, dimRoundingMode);
        var res = environment_1.ENV.engine.executeKernel('Conv2DDerInput', { inputs: { dy: dy4D, filter: filter }, args: { convInfo: convInfo } });
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    Ops.conv2dDerBias = function (dy) {
        var dy4D = dy;
        if (dy.rank === 3) {
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        return environment_1.ENV.engine.executeKernel('Conv2DDerBias', { inputs: { dy: dy4D } });
    };
    Ops.conv2dDerFilter = function (x, dy, filterShape, strides, pad, dimRoundingMode) {
        var x4D = x;
        if (x.rank === 3) {
            x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        }
        var dy4D = dy;
        if (dy4D.rank === 3) {
            dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
        }
        util.assert(x4D.rank === 4, "Error in conv2dDerFilter: input must be rank 4, but got shape " +
            (x4D.shape + "."));
        util.assert(dy4D.rank === 4, "Error in conv2dDerFilter: dy must be rank 4, but got shape " +
            (dy4D.shape + "."));
        util.assert(filterShape.length === 4, "Error in conv2dDerFilter: filterShape must be length 4, but got " +
            (filterShape + "."));
        util.assert(x4D.shape[3] === filterShape[2], "Error in conv2dDerFilter: depth of input " + x4D.shape[3] + ") must " +
            ("match input depth in filter (" + filterShape[2] + "."));
        util.assert(dy4D.shape[3] === filterShape[3], "Error in conv2dDerFilter: depth of dy (" + dy4D.shape[3] + ") must " +
            ("match output depth for filter (" + filterShape[3] + ")."));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in conv2dDerFilter: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computeConv2DInfo(x4D.shape, filterShape, strides, pad, dimRoundingMode);
        return environment_1.ENV.engine.executeKernel('Conv2DDerFilter', { inputs: { x: x4D, dy: dy4D }, args: { convInfo: convInfo } });
    };
    Ops.conv2dTranspose = function (x, filter, outputShape, strides, pad, dimRoundingMode) {
        return Ops.conv2dDerInput(outputShape, x, filter, strides, pad, dimRoundingMode);
    };
    Ops.depthwiseConv2D = function (input, filter, strides, pad, rates, dimRoundingMode) {
        if (rates === void 0) { rates = [1, 1]; }
        var input4D = input;
        var reshapedTo4D = false;
        if (input.rank === 3) {
            reshapedTo4D = true;
            input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
        }
        util.assert(input4D.rank === 4, "Error in depthwiseConv2D: input must be rank 4, but got " +
            ("rank " + input4D.rank + "."));
        util.assert(filter.rank === 4, "Error in depthwiseConv2D: filter must be rank 4, but got rank " +
            (filter.rank + "."));
        util.assert(input4D.shape[3] === filter.shape[2], "Error in depthwiseConv2D: number of input channels " +
            ("(" + input4D.shape[3] + ") must match the inChannels dimension in ") +
            ("filter " + filter.shape[2] + "."));
        rates = rates || [1, 1];
        var _a = parseTupleParam(rates), rateHeight = _a[0], rateWidth = _a[1];
        util.assert(rateHeight === 1 && rateWidth === 1, 'Error in depthwiseConv2D: rates greater than 1 are not yet ' +
            ("supported. Got rates '" + rates + "'"));
        if (dimRoundingMode != null) {
            util.assert(util.isInt(pad), "Error in depthwiseConv2D: pad must be an integer when using, " +
                ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
        }
        var convInfo = conv_util.computeConv2DInfo(input4D.shape, filter.shape, strides, pad, dimRoundingMode, true);
        var res = environment_1.ENV.engine.executeKernel('DepthwiseConv2D', { inputs: { x: input4D, filter: filter }, args: { convInfo: convInfo } });
        if (reshapedTo4D) {
            return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
        }
        return res;
    };
    __decorate([
        decorators_1.operation
    ], Ops, "conv1d", null);
    __decorate([
        decorators_1.operation
    ], Ops, "conv2d", null);
    __decorate([
        decorators_1.operation
    ], Ops, "conv2dDerInput", null);
    __decorate([
        decorators_1.operation
    ], Ops, "conv2dDerBias", null);
    __decorate([
        decorators_1.operation
    ], Ops, "conv2dDerFilter", null);
    __decorate([
        decorators_1.operation
    ], Ops, "conv2dTranspose", null);
    __decorate([
        decorators_1.operation
    ], Ops, "depthwiseConv2D", null);
    return Ops;
}());
exports.Ops = Ops;
function parseTupleParam(param) {
    return typeof param === 'number' ? [param, param] : param;
}
//# sourceMappingURL=conv.js.map