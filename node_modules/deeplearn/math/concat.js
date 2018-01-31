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
var concat_util = require("./concat_util");
var decorators_1 = require("./decorators");
var Ops = (function () {
    function Ops() {
    }
    Ops.concat1D = function (a, b) {
        return Ops.concat(a, b, 0);
    };
    Ops.concat2D = function (a, b, axis) {
        return Ops.concat(a, b, axis);
    };
    Ops.concat3D = function (a, b, axis) {
        return Ops.concat(a, b, axis);
    };
    Ops.concat4D = function (a, b, axis) {
        return Ops.concat(a, b, axis);
    };
    Ops.concat = function (a, b, axis) {
        concat_util.assertParams(a.shape, b.shape, axis);
        var outShape = concat_util.computeOutShape(a.shape, b.shape, axis);
        var a2D = a.as2D(-1, util.sizeFromShape(a.shape.slice(axis)));
        var b2D = b.as2D(-1, util.sizeFromShape(b.shape.slice(axis)));
        var _a = concat_util.computeGradientSliceShapes(a2D.shape, b2D.shape), aBegin = _a.aBegin, aSize = _a.aSize, bBegin = _a.bBegin, bSize = _a.bSize;
        var der = function (dy) {
            return {
                a: function () { return dy.slice(aBegin, aSize); },
                b: function () { return dy.slice(bBegin, bSize); }
            };
        };
        var res = environment_1.ENV.engine.executeKernel('Concat', { inputs: { a: a2D, b: b2D } }, der);
        return res.reshape(outShape);
    };
    __decorate([
        decorators_1.operation
    ], Ops, "concat1D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "concat2D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "concat3D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "concat4D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "concat", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=concat.js.map