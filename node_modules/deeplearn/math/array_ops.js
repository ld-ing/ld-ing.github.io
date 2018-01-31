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
var ndarray_1 = require("./ndarray");
var rand_1 = require("./rand");
var Ops = (function () {
    function Ops() {
    }
    Ops.ones = function (shape, dtype) {
        var values = makeOnesTypedArray(util.sizeFromShape(shape), dtype);
        return ndarray_1.NDArray.make(shape, { values: values }, dtype);
    };
    Ops.zeros = function (shape, dtype) {
        var values = makeZerosTypedArray(util.sizeFromShape(shape), dtype);
        return ndarray_1.NDArray.make(shape, { values: values }, dtype);
    };
    Ops.onesLike = function (x) {
        return Ops.ones(x.shape, x.dtype);
    };
    Ops.zerosLike = function (x) {
        return Ops.zeros(x.shape, x.dtype);
    };
    Ops.clone = function (x) {
        var newValues = util.copyTypedArray(x.dataSync(), x.dtype);
        return ndarray_1.NDArray.make(x.shape, { values: newValues }, x.dtype);
    };
    Ops.randNormal = function (shape, mean, stdDev, dtype, seed) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        if (dtype != null && dtype === 'bool') {
            throw new Error("Unsupported data type " + dtype);
        }
        var randGauss = new rand_1.MPRandGauss(mean, stdDev, dtype, false, seed);
        return ndarray_1.NDArray.rand(shape, function () { return randGauss.nextValue(); }, dtype);
    };
    Ops.truncatedNormal = function (shape, mean, stdDev, dtype, seed) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        if (dtype != null && dtype === 'bool') {
            throw new Error("Unsupported data type " + dtype);
        }
        var randGauss = new rand_1.MPRandGauss(mean, stdDev, dtype, true, seed);
        return ndarray_1.NDArray.rand(shape, function () { return randGauss.nextValue(); }, dtype);
    };
    Ops.randUniform = function (shape, a, b, dtype) {
        return ndarray_1.NDArray.rand(shape, function () { return util.randUniform(a, b); }, dtype);
    };
    Ops.rand = function (shape, randFunction, dtype) {
        var size = util.sizeFromShape(shape);
        var values = null;
        if (dtype == null || dtype === 'float32') {
            values = new Float32Array(size);
        }
        else if (dtype === 'int32') {
            values = new Int32Array(size);
        }
        else if (dtype === 'bool') {
            values = new Uint8Array(size);
        }
        else {
            throw new Error("Unknown data type " + dtype);
        }
        for (var i = 0; i < size; i++) {
            values[i] = randFunction();
        }
        return ndarray_1.NDArray.make(shape, { values: values }, dtype);
    };
    Ops.multinomial = function (probabilities, numSamples, seed) {
        var numOutcomes = probabilities.size;
        if (numOutcomes < 2) {
            throw new Error("Error in multinomial: you need at least 2 outcomes, but got " +
                (numOutcomes + "."));
        }
        if (probabilities.rank > 2) {
            throw new Error("Rank of probabilities must be 1 or 2, but is " + probabilities.rank);
        }
        seed = seed || Math.random();
        var origRank = probabilities.rank;
        if (probabilities.rank === 1) {
            probabilities = probabilities.as2D(1, -1);
        }
        var res = environment_1.ENV.engine.executeKernel('Multinomial', {
            inputs: { probs: probabilities },
            args: { numSamples: numSamples, seed: seed }
        });
        if (origRank === 1) {
            return res.as1D();
        }
        return res;
    };
    Ops.oneHot = function (indices, depth, onValue, offValue) {
        if (onValue === void 0) { onValue = 1; }
        if (offValue === void 0) { offValue = 0; }
        if (depth < 2) {
            throw new Error("Error in oneHot: depth must be >=2, but it is " + depth);
        }
        return environment_1.ENV.engine.executeKernel('OneHot', { inputs: { indices: indices }, args: { depth: depth, onValue: onValue, offValue: offValue } });
    };
    Ops.fromPixels = function (pixels, numChannels) {
        if (numChannels === void 0) { numChannels = 3; }
        if (numChannels > 4) {
            throw new Error('Cannot construct NDArray with more than 4 channels from pixels.');
        }
        return environment_1.ENV.math.fromPixels(pixels, numChannels);
    };
    Ops.reshape = function (x, newShape) {
        newShape = util.inferFromImplicitShape(newShape, x.size);
        util.assert(x.size === util.sizeFromShape(newShape), 'new shape and old shape must have the same number of elements.');
        var grad = function (dy, y) {
            return { x: function () { return dy.reshape(x.shape); } };
        };
        return environment_1.ENV.engine.executeKernel('Reshape', { inputs: { x: x }, args: { newShape: newShape } }, grad);
    };
    Ops.cast = function (x, newDType) {
        var grad = function (dy, y) {
            return { x: function () { return dy.reshape(dy.shape); } };
        };
        return environment_1.ENV.engine.executeKernel('Cast', { inputs: { x: x }, args: { newDType: newDType } }, grad);
    };
    Ops.tile = function (x, reps) {
        util.assert(x.rank === reps.length, "Error in transpose: rank of input " + x.rank + " " +
            ("must match length of reps " + reps + "."));
        return environment_1.ENV.engine.executeKernel('Tile', { inputs: { x: x }, args: { reps: reps } });
    };
    Ops.gather = function (x, indices, axis) {
        if (axis === void 0) { axis = 0; }
        return environment_1.ENV.engine.executeKernel('Gather', { inputs: { x: x, indices: indices }, args: { axis: axis } });
    };
    Ops.pad1D = function (x, paddings, constantValue) {
        if (constantValue === void 0) { constantValue = 0; }
        util.assert(paddings.length === 2, 'Invalid number of paddings. Must be length of 2.');
        return environment_1.ENV.engine.executeKernel('Pad1D', { inputs: { x: x }, args: { paddings: paddings, constantValue: constantValue } });
    };
    Ops.pad2D = function (x, paddings, constantValue) {
        if (constantValue === void 0) { constantValue = 0; }
        util.assert(paddings.length === 2 && paddings[0].length === 2 &&
            paddings[1].length === 2, 'Invalid number of paddings. Must be length of 2 each.');
        return environment_1.ENV.engine.executeKernel('Pad2D', { inputs: { x: x }, args: { paddings: paddings, constantValue: constantValue } });
    };
    __decorate([
        decorators_1.operation
    ], Ops, "ones", null);
    __decorate([
        decorators_1.operation
    ], Ops, "zeros", null);
    __decorate([
        decorators_1.operation
    ], Ops, "onesLike", null);
    __decorate([
        decorators_1.operation
    ], Ops, "zerosLike", null);
    __decorate([
        decorators_1.operation
    ], Ops, "clone", null);
    __decorate([
        decorators_1.operation
    ], Ops, "randNormal", null);
    __decorate([
        decorators_1.operation
    ], Ops, "truncatedNormal", null);
    __decorate([
        decorators_1.operation
    ], Ops, "randUniform", null);
    __decorate([
        decorators_1.operation
    ], Ops, "rand", null);
    __decorate([
        decorators_1.operation
    ], Ops, "multinomial", null);
    __decorate([
        decorators_1.operation
    ], Ops, "oneHot", null);
    __decorate([
        decorators_1.operation
    ], Ops, "fromPixels", null);
    __decorate([
        decorators_1.operation
    ], Ops, "reshape", null);
    __decorate([
        decorators_1.operation
    ], Ops, "cast", null);
    __decorate([
        decorators_1.operation
    ], Ops, "tile", null);
    __decorate([
        decorators_1.operation
    ], Ops, "gather", null);
    __decorate([
        decorators_1.operation
    ], Ops, "pad1D", null);
    __decorate([
        decorators_1.operation
    ], Ops, "pad2D", null);
    return Ops;
}());
exports.Ops = Ops;
function makeZerosTypedArray(size, dtype) {
    if (dtype == null || dtype === 'float32') {
        return new Float32Array(size);
    }
    else if (dtype === 'int32') {
        return new Int32Array(size);
    }
    else if (dtype === 'bool') {
        return new Uint8Array(size);
    }
    else {
        throw new Error("Unknown data type " + dtype);
    }
}
function makeOnesTypedArray(size, dtype) {
    var array = makeZerosTypedArray(size, dtype);
    for (var i = 0; i < array.length; i++) {
        array[i] = 1;
    }
    return array;
}
//# sourceMappingURL=array_ops.js.map