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
    Ops.transpose = function (x, perm) {
        if (perm == null) {
            perm = x.shape.map(function (s, i) { return i; }).reverse();
        }
        var der = function (dy) {
            var undoPerm = axis_util.getUndoAxesPermutation(perm);
            var derX = function () { return dy.transpose(undoPerm); };
            return { x: derX };
        };
        util.assert(x.rank === perm.length, "Error in transpose: rank of input " + x.rank + " " +
            ("must match length of perm " + perm + "."));
        return environment_1.ENV.engine.executeKernel('Transpose', { inputs: { x: x }, args: { perm: perm } }, der);
    };
    __decorate([
        decorators_1.operation
    ], Ops, "transpose", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=transpose.js.map