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
var Ops = (function () {
    function Ops() {
    }
    Ops.reverse1D = function (x) {
        util.assert(x.rank === 1, "Error in reverse1D: x must be rank 1 but got\n             rank " + x.rank + ".");
        var input4D = x.as4D(1, 1, 1, x.shape[0]);
        var res = Ops.reverse4D(input4D, [3]);
        return res.as1D();
    };
    Ops.reverse2D = function (x, axis) {
        util.assert(x.rank === 2, "Error in reverse2D: x must be rank 2 but got\n             rank " + x.rank + ".");
        var axisCleaned = axis_util.parseAxisParam(axis, x.shape).map(function (a) { return a + 2; });
        var input4D = x.as4D(1, 1, x.shape[0], x.shape[1]);
        var res = Ops.reverse4D(input4D, axisCleaned);
        return res.as2D(res.shape[2], res.shape[3]);
    };
    Ops.reverse3D = function (x, axis) {
        util.assert(x.rank === 3, "Error in reverse3D: x must be rank 3 but got\n             rank " + x.rank + ".");
        var axisCleaned = axis_util.parseAxisParam(axis, x.shape).map(function (a) { return a + 1; });
        var input4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        var res = Ops.reverse4D(input4D, axisCleaned);
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    };
    Ops.reverse4D = function (x, axis) {
        util.assert(x.rank === 4, "Error in reverse4D: x must be rank 4 but got\n             rank " + x.rank + ".");
        var axisCleaned = axis_util.parseAxisParam(axis, x.shape);
        return environment_1.ENV.engine.executeKernel('Reverse4D', { inputs: { x: x }, args: { axis: axisCleaned } });
    };
    Ops.reverse = function (x, axis) {
        if (x.rank === 0) {
            return x.reshape(x.shape);
        }
        else if (x.rank === 1) {
            return Ops.reverse1D(x);
        }
        else if (x.rank === 2) {
            return Ops.reverse2D(x, axis);
        }
        else if (x.rank === 3) {
            return Ops.reverse3D(x, axis);
        }
        else if (x.rank === 4) {
            return Ops.reverse4D(x, axis);
        }
        else {
            throw new Error("Reverse for rank " + x.rank + " is not yet implemented");
        }
    };
    __decorate([
        decorators_1.operation
    ], Ops, "reverse1D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "reverse2D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "reverse3D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "reverse4D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "reverse", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=reverse.js.map