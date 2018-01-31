"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var util = require("../util");
var matmul_1 = require("./backends/types/matmul");
var ops = require("./ops");
var NDArray = (function () {
    function NDArray(shape, dtype, values, dataId) {
        this.isDisposed = false;
        this.size = util.sizeFromShape(shape);
        if (values != null) {
            util.assert(this.size === values.length, "Constructing ndarray of shape (" + this.size + ") should match the " +
                ("length of values (" + values.length + ")"));
        }
        this.shape = shape;
        this.dtype = dtype || 'float32';
        var dim = this.shape.length;
        if (dim < 2) {
            this.strides = [];
        }
        else {
            this.strides = new Array(dim - 1);
            this.strides[dim - 2] = this.shape[dim - 1];
            for (var i = dim - 3; i >= 0; --i) {
                this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
            }
        }
        this.dataId = dataId != null ? dataId : NDArray.nextDataId++;
        this.id = NDArray.nextId++;
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
        environment_1.ENV.math.register(this);
        if (values != null) {
            environment_1.ENV.math.write(this.dataId, values);
        }
    }
    NDArray.ones = function (shape, dtype) {
        return ops.ones(shape, dtype);
    };
    NDArray.zeros = function (shape, dtype) {
        return ops.zeros(shape, dtype);
    };
    NDArray.onesLike = function (x) {
        return ops.onesLike(x);
    };
    NDArray.zerosLike = function (x) {
        return ops.zerosLike(x);
    };
    NDArray.like = function (x) {
        return ops.clone(x);
    };
    NDArray.make = function (shape, data, dtype) {
        return new NDArray(shape, dtype, data.values, data.dataId);
    };
    NDArray.fromPixels = function (pixels, numChannels) {
        if (numChannels === void 0) { numChannels = 3; }
        return ops.fromPixels(pixels, numChannels);
    };
    NDArray.rand = function (shape, randFunction, dtype) {
        return ops.rand(shape, randFunction, dtype);
    };
    NDArray.randNormal = function (shape, mean, stdDev, dtype, seed) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return ops.randNormal(shape, mean, stdDev, dtype, seed);
    };
    NDArray.randTruncatedNormal = function (shape, mean, stdDev, dtype, seed) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return ops.truncatedNormal(shape, mean, stdDev, dtype, seed);
    };
    NDArray.randUniform = function (shape, a, b, dtype) {
        return ops.randUniform(shape, a, b, dtype);
    };
    NDArray.prototype.squeeze = function (axis) {
        this.throwIfDisposed();
        return this.reshape(util.squeezeShape(this.shape, axis).newShape);
    };
    NDArray.prototype.flatten = function () {
        this.throwIfDisposed();
        return this.as1D();
    };
    NDArray.prototype.asScalar = function () {
        this.throwIfDisposed();
        util.assert(this.size === 1, 'The array must have only 1 element.');
        return this.reshape([]);
    };
    NDArray.prototype.as1D = function () {
        this.throwIfDisposed();
        return this.reshape([this.size]);
    };
    NDArray.prototype.as2D = function (rows, columns) {
        this.throwIfDisposed();
        return this.reshape([rows, columns]);
    };
    NDArray.prototype.as3D = function (rows, columns, depth) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth]);
    };
    NDArray.prototype.as4D = function (rows, columns, depth, depth2) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth, depth2]);
    };
    NDArray.prototype.asType = function (dtype) {
        this.throwIfDisposed();
        return ops.cast(this, dtype);
    };
    Object.defineProperty(NDArray.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    NDArray.prototype.get = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        this.throwIfDisposed();
        if (locs.length === 0) {
            locs = [0];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.dataSync()[index];
    };
    NDArray.prototype.set = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        if (locs.length === 0) {
            locs = [0];
        }
        this.throwIfDisposed();
        util.assert(locs.length === this.rank, "The number of provided coordinates (" + locs.length + ") must " +
            ("match the rank (" + this.rank + ")"));
        var index = locs.length > 0 ? locs[locs.length - 1] : 0;
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        var vals = this.dataSync();
        vals[index] = value;
        environment_1.ENV.math.write(this.dataId, vals);
    };
    NDArray.prototype.val = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (locs.length === 0) {
                            locs = [0];
                        }
                        this.throwIfDisposed();
                        return [4, this.data()];
                    case 1:
                        _a.sent();
                        return [2, this.get.apply(this, locs)];
                }
            });
        });
    };
    NDArray.prototype.locToIndex = function (locs) {
        this.throwIfDisposed();
        if (this.rank === 0) {
            return 0;
        }
        else if (this.rank === 1) {
            return locs[0];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    };
    NDArray.prototype.indexToLoc = function (index) {
        this.throwIfDisposed();
        if (this.rank === 0) {
            return [];
        }
        else if (this.rank === 1) {
            return [index];
        }
        var locs = new Array(this.shape.length);
        for (var i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    };
    NDArray.prototype.fill = function (value) {
        this.throwIfDisposed();
        var vals = this.dataSync();
        vals.fill(value);
        environment_1.ENV.math.write(this.dataId, vals);
    };
    NDArray.prototype.getValues = function () {
        return this.dataSync();
    };
    NDArray.prototype.getValuesAsync = function () {
        return this.data();
    };
    NDArray.prototype.data = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                this.throwIfDisposed();
                return [2, environment_1.ENV.math.read(this.dataId)];
            });
        });
    };
    NDArray.prototype.dataSync = function () {
        this.throwIfDisposed();
        return environment_1.ENV.math.readSync(this.dataId);
    };
    NDArray.prototype.dispose = function () {
        if (this.isDisposed) {
            return;
        }
        this.isDisposed = true;
        environment_1.ENV.math.disposeData(this.dataId);
    };
    NDArray.prototype.throwIfDisposed = function () {
        if (this.isDisposed) {
            throw new Error("NDArray is disposed.");
        }
    };
    NDArray.prototype.toFloat = function () {
        return this.asType('float32');
    };
    NDArray.prototype.toInt = function () {
        return this.asType('int32');
    };
    NDArray.prototype.toBool = function () {
        return this.asType('bool');
    };
    NDArray.prototype.reshape = function (newShape) {
        this.throwIfDisposed();
        return ops.reshape(this, newShape);
    };
    NDArray.prototype.reshapeAs = function (x) {
        this.throwIfDisposed();
        return this.reshape(x.shape);
    };
    NDArray.prototype.tile = function (reps) {
        this.throwIfDisposed();
        return ops.tile(this, reps);
    };
    NDArray.prototype.gather = function (indices, axis) {
        if (axis === void 0) { axis = 0; }
        this.throwIfDisposed();
        return ops.gather(this, indices);
    };
    NDArray.prototype.matMul = function (b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = matmul_1.MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = matmul_1.MatrixOrientation.REGULAR; }
        this.throwIfDisposed();
        return ops.matMul(this, b, aOrientation, bOrientation);
    };
    NDArray.prototype.slice = function (begin, size) {
        this.throwIfDisposed();
        return ops.slice(this, begin, size);
    };
    NDArray.prototype.reverse = function (axis) {
        this.throwIfDisposed();
        return ops.reverse(this, axis);
    };
    NDArray.prototype.concat = function (x, axis) {
        this.throwIfDisposed();
        return ops.concat(this, x, axis);
    };
    NDArray.prototype.batchNormalization = function (mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        this.throwIfDisposed();
        return ops.batchNormalization(this, mean, variance, varianceEpsilon, scale, offset);
    };
    NDArray.prototype.clone = function () {
        this.throwIfDisposed();
        return ops.clone(this);
    };
    NDArray.prototype.logSumExp = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return ops.logSumExp(this, axis, keepDims);
    };
    NDArray.prototype.sum = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return ops.sum(this, axis, keepDims);
    };
    NDArray.prototype.mean = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return ops.mean(this, axis, keepDims);
    };
    NDArray.prototype.min = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return ops.min(this, axis, keepDims);
    };
    NDArray.prototype.max = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return ops.max(this, axis, keepDims);
    };
    NDArray.prototype.argMin = function (axis) {
        if (axis === void 0) { axis = null; }
        this.throwIfDisposed();
        return ops.argMin(this, axis);
    };
    NDArray.prototype.argMax = function (axis) {
        if (axis === void 0) { axis = null; }
        this.throwIfDisposed();
        return ops.argMax(this, axis);
    };
    NDArray.prototype.argMaxEquals = function (x) {
        this.throwIfDisposed();
        return ops.argMaxEquals(this, x);
    };
    NDArray.prototype.add = function (x) {
        this.throwIfDisposed();
        return ops.add(this, x);
    };
    NDArray.prototype.addStrict = function (x) {
        this.throwIfDisposed();
        return ops.addStrict(this, x);
    };
    NDArray.prototype.sub = function (x) {
        this.throwIfDisposed();
        return ops.sub(this, x);
    };
    NDArray.prototype.subStrict = function (x) {
        this.throwIfDisposed();
        return ops.subStrict(this, x);
    };
    NDArray.prototype.pow = function (exp) {
        this.throwIfDisposed();
        return ops.pow(this, exp);
    };
    NDArray.prototype.powStrict = function (exp) {
        this.throwIfDisposed();
        return ops.powStrict(this, exp);
    };
    NDArray.prototype.mul = function (x) {
        this.throwIfDisposed();
        return ops.mul(this, x);
    };
    NDArray.prototype.mulStrict = function (x) {
        this.throwIfDisposed();
        return ops.mulStrict(this, x);
    };
    NDArray.prototype.div = function (x) {
        this.throwIfDisposed();
        return ops.div(this, x);
    };
    NDArray.prototype.divStrict = function (x) {
        this.throwIfDisposed();
        return ops.divStrict(this, x);
    };
    NDArray.prototype.minimum = function (x) {
        this.throwIfDisposed();
        return ops.minimum(this, x);
    };
    NDArray.prototype.minimumStrict = function (x) {
        this.throwIfDisposed();
        return ops.minimumStrict(this, x);
    };
    NDArray.prototype.maximum = function (x) {
        this.throwIfDisposed();
        return ops.maximum(this, x);
    };
    NDArray.prototype.maximumStrict = function (x) {
        this.throwIfDisposed();
        return ops.maximumStrict(this, x);
    };
    NDArray.prototype.transpose = function (perm) {
        this.throwIfDisposed();
        return ops.transpose(this, perm);
    };
    NDArray.prototype.notEqual = function (x) {
        this.throwIfDisposed();
        return ops.notEqual(this, x);
    };
    NDArray.prototype.notEqualStrict = function (x) {
        this.throwIfDisposed();
        return ops.notEqualStrict(this, x);
    };
    NDArray.prototype.less = function (x) {
        this.throwIfDisposed();
        return ops.less(this, x);
    };
    NDArray.prototype.lessStrict = function (x) {
        this.throwIfDisposed();
        return ops.lessStrict(this, x);
    };
    NDArray.prototype.equal = function (x) {
        this.throwIfDisposed();
        return ops.equal(this, x);
    };
    NDArray.prototype.equalStrict = function (x) {
        this.throwIfDisposed();
        return ops.equalStrict(this, x);
    };
    NDArray.prototype.lessEqual = function (x) {
        this.throwIfDisposed();
        return ops.lessEqual(this, x);
    };
    NDArray.prototype.lessEqualStrict = function (x) {
        this.throwIfDisposed();
        return ops.lessEqualStrict(this, x);
    };
    NDArray.prototype.greater = function (x) {
        this.throwIfDisposed();
        return ops.greater(this, x);
    };
    NDArray.prototype.greaterStrict = function (x) {
        this.throwIfDisposed();
        return ops.greaterStrict(this, x);
    };
    NDArray.prototype.greaterEqual = function (x) {
        this.throwIfDisposed();
        return ops.greaterEqual(this, x);
    };
    NDArray.prototype.greaterEqualStrict = function (x) {
        this.throwIfDisposed();
        return ops.greaterEqualStrict(this, x);
    };
    NDArray.prototype.logicalAnd = function (x) {
        this.throwIfDisposed();
        return ops.logicalAnd(this, x);
    };
    NDArray.prototype.logicalOr = function (x) {
        this.throwIfDisposed();
        return ops.logicalOr(this, x);
    };
    NDArray.prototype.where = function (condition, x) {
        this.throwIfDisposed();
        return ops.where(condition, this, x);
    };
    NDArray.prototype.neg = function () {
        this.throwIfDisposed();
        return ops.neg(this);
    };
    NDArray.prototype.ceil = function () {
        this.throwIfDisposed();
        return ops.ceil(this);
    };
    NDArray.prototype.floor = function () {
        this.throwIfDisposed();
        return ops.floor(this);
    };
    NDArray.prototype.exp = function () {
        this.throwIfDisposed();
        return ops.exp(this);
    };
    NDArray.prototype.log = function () {
        this.throwIfDisposed();
        return ops.log(this);
    };
    NDArray.prototype.sqrt = function () {
        this.throwIfDisposed();
        return ops.sqrt(this);
    };
    NDArray.prototype.square = function () {
        this.throwIfDisposed();
        return ops.square(this);
    };
    NDArray.prototype.abs = function () {
        this.throwIfDisposed();
        return ops.abs(this);
    };
    NDArray.prototype.clip = function (min, max) {
        this.throwIfDisposed();
        return ops.clip(this, min, max);
    };
    NDArray.prototype.relu = function () {
        this.throwIfDisposed();
        return ops.relu(this);
    };
    NDArray.prototype.elu = function () {
        this.throwIfDisposed();
        return ops.elu(this);
    };
    NDArray.prototype.selu = function () {
        this.throwIfDisposed();
        return ops.selu(this);
    };
    NDArray.prototype.leakyRelu = function (alpha) {
        if (alpha === void 0) { alpha = 0.2; }
        this.throwIfDisposed();
        return ops.leakyRelu(this, alpha);
    };
    NDArray.prototype.prelu = function (alpha) {
        this.throwIfDisposed();
        return ops.prelu(this, alpha);
    };
    NDArray.prototype.sigmoid = function () {
        this.throwIfDisposed();
        return ops.sigmoid(this);
    };
    NDArray.prototype.sin = function () {
        this.throwIfDisposed();
        return ops.sin(this);
    };
    NDArray.prototype.cos = function () {
        this.throwIfDisposed();
        return ops.cos(this);
    };
    NDArray.prototype.tan = function () {
        this.throwIfDisposed();
        return ops.tan(this);
    };
    NDArray.prototype.asin = function () {
        this.throwIfDisposed();
        return ops.asin(this);
    };
    NDArray.prototype.acos = function () {
        this.throwIfDisposed();
        return ops.acos(this);
    };
    NDArray.prototype.atan = function () {
        this.throwIfDisposed();
        return ops.atan(this);
    };
    NDArray.prototype.sinh = function () {
        this.throwIfDisposed();
        return ops.sinh(this);
    };
    NDArray.prototype.cosh = function () {
        this.throwIfDisposed();
        return ops.cosh(this);
    };
    NDArray.prototype.tanh = function () {
        this.throwIfDisposed();
        return ops.tanh(this);
    };
    NDArray.prototype.step = function (alpha) {
        if (alpha === void 0) { alpha = 0.0; }
        this.throwIfDisposed();
        return ops.step(this, alpha);
    };
    NDArray.prototype.softmax = function (dim) {
        if (dim === void 0) { dim = -1; }
        this.throwIfDisposed();
        return ops.softmax(this, dim);
    };
    NDArray.prototype.resizeBilinear = function (newShape2D, alignCorners) {
        if (alignCorners === void 0) { alignCorners = false; }
        this.throwIfDisposed();
        return ops.image.resizeBilinear(this, newShape2D, alignCorners);
    };
    NDArray.prototype.conv1d = function (filter, bias, stride, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return ops.conv1d(this, filter, bias, stride, pad, dimRoundingMode);
    };
    NDArray.prototype.conv2d = function (filter, bias, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return ops.conv2d(this, filter, bias, strides, pad, dimRoundingMode);
    };
    NDArray.prototype.conv2dTranspose = function (filter, outputShape, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return ops.conv2dTranspose(this, filter, outputShape, strides, pad, dimRoundingMode);
    };
    NDArray.prototype.depthwiseConv2D = function (filter, strides, pad, rates, dimRoundingMode) {
        if (rates === void 0) { rates = [1, 1]; }
        this.throwIfDisposed();
        return ops.depthwiseConv2D(this, filter, strides, pad, rates, dimRoundingMode);
    };
    NDArray.prototype.avgPool = function (filterSize, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return ops.avgPool(this, filterSize, strides, pad, dimRoundingMode);
    };
    NDArray.prototype.maxPool = function (filterSize, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return ops.maxPool(this, filterSize, strides, pad, dimRoundingMode);
    };
    NDArray.prototype.minPool = function (filterSize, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return ops.minPool(this, filterSize, strides, pad, dimRoundingMode);
    };
    NDArray.nextId = 0;
    NDArray.nextDataId = 0;
    return NDArray;
}());
exports.NDArray = NDArray;
var Scalar = (function (_super) {
    __extends(Scalar, _super);
    function Scalar() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Scalar.new = function (value, dtype) {
        var values = [value];
        return new Scalar([], dtype, toTypedArray(values, dtype));
    };
    return Scalar;
}(NDArray));
exports.Scalar = Scalar;
var Array1D = (function (_super) {
    __extends(Array1D, _super);
    function Array1D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Array1D.new = function (values, dtype) {
        if (!instanceofTypedArray(values)) {
            var inferredShape = util.inferShape(values);
            util.assert(inferredShape.length === 1, "Error constructing Array1D. Shape of values " + inferredShape + " is " +
                "not 1 dimensional.");
        }
        return new Array1D([values.length], dtype, toTypedArray(values, dtype));
    };
    return Array1D;
}(NDArray));
exports.Array1D = Array1D;
var Array2D = (function (_super) {
    __extends(Array2D, _super);
    function Array2D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Array2D.new = function (shape, values, dtype) {
        if (!instanceofTypedArray(values)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array2D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array2D(shape, dtype, toTypedArray(values, dtype));
    };
    return Array2D;
}(NDArray));
exports.Array2D = Array2D;
var Array3D = (function (_super) {
    __extends(Array3D, _super);
    function Array3D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Array3D.new = function (shape, values, dtype) {
        if (!instanceofTypedArray(values)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array3D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array3D(shape, dtype, toTypedArray(values, dtype));
    };
    return Array3D;
}(NDArray));
exports.Array3D = Array3D;
var Array4D = (function (_super) {
    __extends(Array4D, _super);
    function Array4D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Array4D.new = function (shape, values, dtype) {
        if (!instanceofTypedArray(values)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array4D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array4D(shape, dtype, toTypedArray(values, dtype));
    };
    return Array4D;
}(NDArray));
exports.Array4D = Array4D;
var Variable = (function (_super) {
    __extends(Variable, _super);
    function Variable(initialValue, trainable, name) {
        if (trainable === void 0) { trainable = true; }
        var _this = _super.call(this, initialValue.shape, initialValue.dtype, null, initialValue.dataId) || this;
        _this.trainable = trainable;
        initialValue.dispose();
        _this.name = name;
        if (_this.name == null) {
            _this.name = Variable.nextVarId.toString();
            Variable.nextVarId++;
        }
        environment_1.ENV.math.registerVariable(_this);
        return _this;
    }
    Variable.variable = function (initialValue, trainable, name, dtype) {
        if (trainable === void 0) { trainable = true; }
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.asType(dtype);
        }
        return new Variable(initialValue, trainable, name);
    };
    Variable.prototype.assign = function (newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error("dtype of the new value (" + newValue.dtype + ") and " +
                ("previous value (" + this.dtype + ") must match"));
        }
        if (!util.arraysEqual(newValue.shape, this.shape)) {
            throw new Error("shape of the new value (" + newValue.shape + ") and " +
                ("previous value (" + this.shape + ") must match"));
        }
        environment_1.ENV.math.disposeData(this.dataId);
        this.dataId = newValue.dataId;
        environment_1.ENV.math.register(this);
        newValue.dispose();
    };
    Variable.nextVarId = 0;
    return Variable;
}(NDArray));
exports.Variable = Variable;
var variable = Variable.variable;
exports.variable = variable;
function instanceofTypedArray(a) {
    return a instanceof Float32Array || a instanceof Int32Array ||
        a instanceof Uint8Array;
}
function noConversionNeeded(a, dtype) {
    return (a instanceof Float32Array && dtype === 'float32') ||
        (a instanceof Int32Array && dtype === 'int32') ||
        (a instanceof Uint8Array && dtype === 'bool');
}
function toTypedArray(a, dtype) {
    if (noConversionNeeded(a, dtype)) {
        return a;
    }
    if (Array.isArray(a)) {
        a = util.flatten(a);
    }
    return util.copyTypedArray(a, dtype);
}
//# sourceMappingURL=ndarray.js.map