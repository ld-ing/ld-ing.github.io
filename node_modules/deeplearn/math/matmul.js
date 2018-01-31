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
var matmul_1 = require("./backends/types/matmul");
var decorators_1 = require("./decorators");
var Ops = (function () {
    function Ops() {
    }
    Ops.matMul = function (a, b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = matmul_1.MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = matmul_1.MatrixOrientation.REGULAR; }
        var innerShapeA = (aOrientation === matmul_1.MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var innerShapeB = (bOrientation === matmul_1.MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];
        util.assert(a.rank === 2 && b.rank === 2, "Error in matMul: inputs must be rank 2, got ranks " + a.rank +
            (" and " + b.rank + "."));
        util.assert(innerShapeA === innerShapeB, "Error in matMul: inner shapes (" + innerShapeA + ") and (" +
            (innerShapeB + ") of NDArrays with shapes " + a.shape + " and ") +
            (b.shape + " and orientations " + matmul_1.MatrixOrientation[aOrientation]) +
            (" and " + matmul_1.MatrixOrientation[bOrientation] + " must match."));
        return environment_1.ENV.engine.executeKernel('MatMul', { inputs: { a: a, b: b }, args: { aOrientation: aOrientation, bOrientation: bOrientation } }, function (dy, y) {
            if (aOrientation === matmul_1.MatrixOrientation.TRANSPOSED ||
                bOrientation === matmul_1.MatrixOrientation.TRANSPOSED) {
                throw new Error("Backprop for transposed MatMul not yet implemented.");
            }
            return {
                a: function () { return dy.matMul(b.toFloat(), matmul_1.MatrixOrientation.REGULAR, matmul_1.MatrixOrientation.TRANSPOSED); },
                b: function () { return a.toFloat().matMul(dy, matmul_1.MatrixOrientation.TRANSPOSED, matmul_1.MatrixOrientation.REGULAR); }
            };
        });
    };
    Ops.vectorTimesMatrix = function (v, matrix) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: first input must be rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: second input must be rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[0], "Error in vectorTimesMatrix: size of vector (" + v.size + ") " +
            ("must match first dimension of matrix (" + matrix.shape[0] + ")"));
        return v.as2D(1, -1).matMul(matrix).as1D();
    };
    Ops.matrixTimesVector = function (matrix, v) {
        util.assert(v.rank === 1, "Error in matrixTimesVector: second input must rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in matrixTimesVector: first input must be a rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[1], "Error in matrixTimesVector: size of first rank 1 input " + v.size + " " +
            "must match inner dimension of second rank 2 input, but got " +
            ("shape " + matrix.shape + "."));
        return matrix.matMul(v.as2D(-1, 1)).as1D();
    };
    Ops.dotProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in dotProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        util.assert(v1.size === v2.size, "Error in dotProduct: size of inputs (" + v1.size + ") and (" +
            (v2.size + ") must match."));
        return v1.as2D(1, -1).matMul(v2.as2D(-1, 1)).asScalar();
    };
    Ops.outerProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in outerProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        return v1.as2D(-1, 1).matMul(v2.as2D(1, -1));
    };
    __decorate([
        decorators_1.operation
    ], Ops, "matMul", null);
    __decorate([
        decorators_1.operation
    ], Ops, "vectorTimesMatrix", null);
    __decorate([
        decorators_1.operation
    ], Ops, "matrixTimesVector", null);
    __decorate([
        decorators_1.operation
    ], Ops, "dotProduct", null);
    __decorate([
        decorators_1.operation
    ], Ops, "outerProduct", null);
    return Ops;
}());
exports.Ops = Ops;
//# sourceMappingURL=matmul.js.map