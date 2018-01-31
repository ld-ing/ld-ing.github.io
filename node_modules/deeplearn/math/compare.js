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
var broadcast_util = require("./broadcast_util");
var decorators_1 = require("./decorators");
var Ops = (function () {
    function Ops() {
    }
    Ops.notEqual = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return environment_1.ENV.engine.executeKernel('NotEqual', { inputs: { a: a, b: b } });
    };
    Ops.notEqualStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in notEqualStrict: ');
        return a.notEqual(b);
    };
    Ops.less = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return environment_1.ENV.engine.executeKernel('Less', { inputs: { a: a, b: b } });
    };
    Ops.lessStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in lessStrict: ');
        return a.less(b);
    };
    Ops.equal = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return environment_1.ENV.engine.executeKernel('Equal', { inputs: { a: a, b: b } });
    };
    Ops.equalStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in equalStrict: ');
        return a.equal(b);
    };
    Ops.lessEqual = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return environment_1.ENV.engine.executeKernel('LessEqual', { inputs: { a: a, b: b } });
    };
    Ops.lessEqualStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in lessEqualStrict: ');
        return a.lessEqual(b);
    };
    Ops.greater = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return environment_1.ENV.engine.executeKernel('Greater', { inputs: { a: a, b: b } });
    };
    Ops.greaterStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in greaterStrict: ');
        return a.greater(b);
    };
    Ops.greaterEqual = function (a, b) {
        util.assertTypesMatch(a, b);
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        return environment_1.ENV.engine.executeKernel('GreaterEqual', { inputs: { a: a, b: b } });
    };
    Ops.greaterEqualStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in greaterEqualStrict: ');
        return a.greaterEqual(b);
    };
    __decorate([
        decorators_1.operation
    ], Ops, "notEqual", null);
    __decorate([
        decorators_1.operation
    ], Ops, "notEqualStrict", null);
    __decorate([
        decorators_1.operation
    ], Ops, "less", null);
    __decorate([
        decorators_1.operation
    ], Ops, "lessStrict", null);
    __decorate([
        decorators_1.operation
    ], Ops, "equal", null);
    __decorate([
        decorators_1.operation
    ], Ops, "equalStrict", null);
    __decorate([
        decorators_1.operation
    ], Ops, "lessEqual", null);
    __decorate([
        decorators_1.operation
    ], Ops, "lessEqualStrict", null);
    __decorate([
        decorators_1.operation
    ], Ops, "greater", null);
    __decorate([
        decorators_1.operation
    ], Ops, "greaterStrict", null);
    __decorate([
        decorators_1.operation
    ], Ops, "greaterEqual", null);
    __decorate([
        decorators_1.operation
    ], Ops, "greaterEqualStrict", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=compare.js.map