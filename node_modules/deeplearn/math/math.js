"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var util = require("../util");
var array_ops = require("./array_ops");
var backend_engine_1 = require("./backends/backend_engine");
var batchnorm = require("./batchnorm");
var binary_ops = require("./binary_ops");
var compare = require("./compare");
var concat = require("./concat");
var conv = require("./conv");
var image_ops = require("./image_ops");
var logical = require("./logical_ops");
var lstm_ops = require("./lstm");
var matmul = require("./matmul");
var ndarray_1 = require("./ndarray");
var norm = require("./norm");
var ops = require("./ops");
var pool = require("./pool");
var reduction_ops = require("./reduction_ops");
var reverse = require("./reverse");
var slice = require("./slice");
var softmax_ops = require("./softmax");
var transpose = require("./transpose");
var unary_ops = require("./unary_ops");
var NDArrayMath = (function () {
    function NDArrayMath(backend, safeMode) {
        this.registeredArrays = new Map();
        this.customBackend = false;
        this.matMul = matmul.Ops.matMul;
        this.vectorTimesMatrix = matmul.Ops.vectorTimesMatrix;
        this.outerProduct = matmul.Ops.outerProduct;
        this.matrixTimesVector = matmul.Ops.matrixTimesVector;
        this.dotProduct = matmul.Ops.dotProduct;
        this.slice = slice.Ops.slice;
        this.slice1D = slice.Ops.slice1D;
        this.slice2D = slice.Ops.slice2D;
        this.slice3D = slice.Ops.slice3D;
        this.slice4D = slice.Ops.slice4D;
        this.reverse = reverse.Ops.reverse;
        this.reverse1D = reverse.Ops.reverse1D;
        this.reverse2D = reverse.Ops.reverse2D;
        this.reverse3D = reverse.Ops.reverse3D;
        this.reverse4D = reverse.Ops.reverse4D;
        this.concat = concat.Ops.concat;
        this.concat1D = concat.Ops.concat1D;
        this.concat2D = concat.Ops.concat2D;
        this.concat3D = concat.Ops.concat3D;
        this.concat4D = concat.Ops.concat4D;
        this.batchNormalization = batchnorm.Ops.batchNormalization;
        this.batchNormalization2D = batchnorm.Ops.batchNormalization2D;
        this.batchNormalization3D = batchnorm.Ops.batchNormalization3D;
        this.batchNormalization4D = batchnorm.Ops.batchNormalization4D;
        this.avgPool = pool.Ops.avgPool;
        this.maxPool = pool.Ops.maxPool;
        this.minPool = pool.Ops.minPool;
        this.maxPoolBackprop = pool.Ops.maxPoolBackprop;
        this.conv1d = conv.Ops.conv1d;
        this.conv2d = conv.Ops.conv2d;
        this.conv2dTranspose = conv.Ops.conv2dTranspose;
        this.depthwiseConv2D = conv.Ops.depthwiseConv2D;
        this.conv2dDerBias = conv.Ops.conv2dDerBias;
        this.conv2dDerFilter = conv.Ops.conv2dDerFilter;
        this.conv2dDerInput = conv.Ops.conv2dDerInput;
        this.argMax = reduction_ops.Ops.argMax;
        this.argMaxEquals = reduction_ops.Ops.argMaxEquals;
        this.argMin = reduction_ops.Ops.argMin;
        this.logSumExp = reduction_ops.Ops.logSumExp;
        this.max = reduction_ops.Ops.max;
        this.mean = reduction_ops.Ops.mean;
        this.min = reduction_ops.Ops.min;
        this.moments = reduction_ops.Ops.moments;
        this.sum = reduction_ops.Ops.sum;
        this.add = binary_ops.Ops.add;
        this.addStrict = binary_ops.Ops.addStrict;
        this.arrayDividedByScalar = binary_ops.Ops.arrayDividedByScalar;
        this.div = binary_ops.Ops.div;
        this.divide = this.div;
        this.divStrict = binary_ops.Ops.divStrict;
        this.divideStrict = this.divStrict;
        this.elementWiseMul = binary_ops.Ops.elementWiseMul;
        this.maximum = binary_ops.Ops.maximum;
        this.maximumStrict = binary_ops.Ops.maximumStrict;
        this.minimum = binary_ops.Ops.minimum;
        this.minimumStrict = binary_ops.Ops.minimumStrict;
        this.mul = binary_ops.Ops.mul;
        this.multiply = this.mul;
        this.mulStrict = binary_ops.Ops.mulStrict;
        this.multiplyStrict = this.mulStrict;
        this.pow = binary_ops.Ops.pow;
        this.powStrict = binary_ops.Ops.powStrict;
        this.scalarDividedByArray = binary_ops.Ops.scalarDividedByArray;
        this.sub = binary_ops.Ops.sub;
        this.subtract = this.sub;
        this.subStrict = binary_ops.Ops.subStrict;
        this.logicalAnd = logical.Ops.logicalAnd;
        this.logicalOr = logical.Ops.logicalOr;
        this.where = logical.Ops.where;
        this.transpose = transpose.Ops.transpose;
        this.equal = compare.Ops.equal;
        this.equalStrict = compare.Ops.equalStrict;
        this.greater = compare.Ops.greater;
        this.greaterStrict = compare.Ops.greaterStrict;
        this.greaterEqual = compare.Ops.greaterEqual;
        this.greaterEqualStrict = compare.Ops.greaterEqualStrict;
        this.less = compare.Ops.less;
        this.lessStrict = compare.Ops.lessStrict;
        this.lessEqual = compare.Ops.lessEqual;
        this.lessEqualStrict = compare.Ops.lessEqualStrict;
        this.notEqual = compare.Ops.notEqual;
        this.notEqualStrict = compare.Ops.notEqualStrict;
        this.abs = unary_ops.Ops.abs;
        this.acos = unary_ops.Ops.acos;
        this.asin = unary_ops.Ops.asin;
        this.atan = unary_ops.Ops.atan;
        this.ceil = unary_ops.Ops.ceil;
        this.clip = unary_ops.Ops.clip;
        this.cos = unary_ops.Ops.cos;
        this.cosh = unary_ops.Ops.cosh;
        this.elu = unary_ops.Ops.elu;
        this.exp = unary_ops.Ops.exp;
        this.floor = unary_ops.Ops.floor;
        this.leakyRelu = unary_ops.Ops.leakyRelu;
        this.log = unary_ops.Ops.log;
        this.neg = unary_ops.Ops.neg;
        this.prelu = unary_ops.Ops.prelu;
        this.relu = unary_ops.Ops.relu;
        this.selu = unary_ops.Ops.selu;
        this.sigmoid = unary_ops.Ops.sigmoid;
        this.sin = unary_ops.Ops.sin;
        this.sinh = unary_ops.Ops.sinh;
        this.sqrt = unary_ops.Ops.sqrt;
        this.square = unary_ops.Ops.square;
        this.step = unary_ops.Ops.step;
        this.tan = unary_ops.Ops.tan;
        this.tanh = unary_ops.Ops.tanh;
        this.norm = norm.Ops.norm;
        this.basicLSTMCell = lstm_ops.Ops.basicLSTMCell;
        this.multiRNNCell = lstm_ops.Ops.multiRNNCell;
        this.softmax = softmax_ops.Ops.softmax;
        this.softmaxCrossEntropy = softmax_ops.Ops.softmaxCrossEntropy;
        this.cast = array_ops.Ops.cast;
        this.clone = array_ops.Ops.clone;
        this.gather = array_ops.Ops.gather;
        this.reshape = array_ops.Ops.reshape;
        this.tile = array_ops.Ops.tile;
        this.oneHot = array_ops.Ops.oneHot;
        this.multinomial = array_ops.Ops.multinomial;
        this.pad1D = array_ops.Ops.pad1D;
        this.pad2D = array_ops.Ops.pad2D;
        this.resizeBilinear3D = image_ops.Ops.resizeBilinear;
        this.registeredVariables = {};
        if (typeof backend === 'string') {
            this.backend = environment_1.ENV.getBackend(backend);
        }
        else {
            this.customBackend = true;
            this.backend = backend;
        }
        this.engine = new backend_engine_1.BackendEngine(this.backend, safeMode);
        environment_1.ENV.setMath(this);
    }
    NDArrayMath.prototype.time = function (query) {
        return this.backend.time(query);
    };
    NDArrayMath.prototype.getNumArrays = function () {
        return this.registeredArrays.size;
    };
    NDArrayMath.prototype.register = function (a) {
        var refCount = this.registeredArrays.has(a.dataId) ?
            this.registeredArrays.get(a.dataId) :
            0;
        if (refCount === 0) {
            this.backend.register(a.dataId, a.shape, a.dtype);
        }
        this.registeredArrays.set(a.dataId, refCount + 1);
        if (!(a instanceof ndarray_1.Variable)) {
            this.engine.track(a);
        }
    };
    NDArrayMath.prototype.registerVariable = function (v) {
        if (this.registeredVariables[v.name] != null) {
            throw new Error("Variable with name " + v.name + " was already registered");
        }
        this.registeredVariables[v.name] = v;
    };
    NDArrayMath.prototype.fromPixels = function (pixels, numChannels) {
        return this.backend.fromPixels(pixels, numChannels);
    };
    NDArrayMath.prototype.write = function (dataId, values) {
        this.backend.write(dataId, values);
    };
    NDArrayMath.prototype.readSync = function (dataId) {
        return this.backend.readSync(dataId);
    };
    NDArrayMath.prototype.read = function (dataId) {
        return this.backend.read(dataId);
    };
    NDArrayMath.prototype.enableDebugMode = function () {
        this.engine.enableDebugMode();
        console.warn('Debugging mode is ON. The output of every math call will ' +
            'be downloaded to CPU and checked for NaNs. ' +
            'This significantly impacts performance.');
    };
    NDArrayMath.prototype.scope = function (nameOrScopeFn, scopeFn, gradientsMode) {
        if (gradientsMode === void 0) { gradientsMode = false; }
        if (scopeFn == null) {
            if (typeof nameOrScopeFn !== 'function') {
                throw new Error('Please provide a function to math.scope()');
            }
            scopeFn = nameOrScopeFn;
            nameOrScopeFn = 'scope';
        }
        else {
            if (typeof nameOrScopeFn !== 'string' &&
                !(nameOrScopeFn instanceof String)) {
                throw new Error('When calling with two arguments, the first argument ' +
                    'to math.scope() must be a string');
            }
            if (typeof scopeFn !== 'function') {
                throw new Error('When calling with two arguments, the 2nd argument ' +
                    'to math.scope() must be a function');
            }
        }
        return this.engine.scope(nameOrScopeFn, scopeFn, gradientsMode);
    };
    NDArrayMath.prototype.gradientsScope = function (nameOrScopeFn, scopeFn) {
        var gradientsMode = true;
        return this.scope(nameOrScopeFn, scopeFn, gradientsMode);
    };
    NDArrayMath.prototype.startScope = function () {
        var gradientsMode = false;
        this.engine.startScope(gradientsMode);
    };
    NDArrayMath.prototype.endScope = function (result) {
        var gradientsMode = false;
        this.engine.endScope(result, gradientsMode);
    };
    NDArrayMath.prototype.keep = function (result) {
        return this.engine.keep(result);
    };
    NDArrayMath.prototype.track = function (result) {
        return result;
    };
    NDArrayMath.prototype.dispose = function () {
        if (this.customBackend) {
            this.backend.dispose();
        }
    };
    NDArrayMath.prototype.topK = function (x, k) {
        var _this = this;
        util.assert(k <= x.size, "Error in topK: k value (" + k + ") must be less than size of input " +
            ("ndarray, got shape " + x.shape + "."));
        var values;
        var indices;
        this.scope('topK', function () {
            values =
                _this.engine.executeKernel('TopKValues', { inputs: { x: x }, args: { k: k } });
            indices =
                _this.engine.executeKernel('TopKIndices', { inputs: { x: x }, args: { k: k } });
            return values;
        });
        var result = { values: values, indices: indices };
        return result;
    };
    NDArrayMath.prototype.switchDim = function (x, perm) {
        return ops.transpose(x, perm);
    };
    NDArrayMath.prototype.scalarPlusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarPlusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.add(c, a);
    };
    NDArrayMath.prototype.scalarMinusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarMinusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.subtract(c, a);
    };
    NDArrayMath.prototype.arrayMinusScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayMinusScalar: second argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.subtract(a, c);
    };
    NDArrayMath.prototype.scaledArrayAdd = function (c1, a, c2, b) {
        var _this = this;
        util.assert(c1.size === 1, "Error in scaledArrayAdd: first argument must rank 0, but got " +
            (" rank " + c1.rank + "."));
        util.assert(c2.size === 1, "Error in scaledArrayAdd: third argument must be rank 0, but got " +
            ("NDArray of rank " + c2.rank + "."));
        util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');
        return this.scope('scaledArrayAdd', function () {
            return _this.add(_this.multiply(c1, a), _this.multiply(c2, b));
        });
    };
    NDArrayMath.prototype.scalarTimesArray = function (c, a) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: first argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.multiply(c, a);
    };
    NDArrayMath.prototype.localResponseNormalization3D = function (x, radius, bias, alpha, beta, normRegion) {
        if (radius === void 0) { radius = 5; }
        if (bias === void 0) { bias = 1; }
        if (alpha === void 0) { alpha = 1; }
        if (beta === void 0) { beta = 0.5; }
        if (normRegion === void 0) { normRegion = 'acrossChannels'; }
        util.assert(x.rank === 3, "Error in localResponseNormalization3D: x must be rank 3 but got\n         rank " + x.rank + ".");
        util.assert(util.isInt(radius), "Error in localResponseNormalization3D: radius must be an integer\n         but got radius " + radius + ".");
        var input4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
        var res = this.localResponseNormalization4D(input4D, radius, bias, alpha, beta, normRegion);
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    };
    NDArrayMath.prototype.localResponseNormalization4D = function (x, radius, bias, alpha, beta, normRegion) {
        if (radius === void 0) { radius = 5; }
        if (bias === void 0) { bias = 1; }
        if (alpha === void 0) { alpha = 1; }
        if (beta === void 0) { beta = 0.5; }
        if (normRegion === void 0) { normRegion = 'acrossChannels'; }
        util.assert(x.rank === 4, "Error in localResponseNormalization4D: x must be rank 4 but got\n         rank " + x.rank + ".");
        util.assert(util.isInt(radius), "Error in localResponseNormalization3D: radius must be an integer\n         but got radius " + radius + ".");
        return this.engine.executeKernel('LRN4D', { inputs: { x: x }, args: { radius: radius, bias: bias, alpha: alpha, beta: beta, normRegion: normRegion } });
    };
    NDArrayMath.prototype.vjp = function (f, x, dy) {
        var keys = x instanceof ndarray_1.NDArray ? null : Object.keys(x);
        var xs = util.flattenNameArrayMap(x, keys);
        var vjp = this.engine.vjp(f, xs, dy);
        if (x instanceof ndarray_1.NDArray) {
            return vjp[0];
        }
        else {
            return util.unflattenToNameArrayMap(keys, vjp);
        }
    };
    NDArrayMath.prototype.gradients = function (f, x) {
        var keys = x instanceof ndarray_1.NDArray ? null : Object.keys(x);
        var xs = util.flattenNameArrayMap(x, keys);
        var returnValue = false;
        var gradients = this.engine.gradients(f, xs, returnValue);
        if (x instanceof ndarray_1.NDArray) {
            return gradients[0];
        }
        else {
            return util.unflattenToNameArrayMap(keys, gradients);
        }
    };
    NDArrayMath.prototype.variableGradients = function (f, varList) {
        if (varList == null) {
            varList = [];
            var varNames = Object.keys(this.registeredVariables);
            for (var i = 0; i < varNames.length; i++) {
                var variable = this.registeredVariables[varNames[i]];
                if (variable.trainable) {
                    varList.push(variable);
                }
            }
        }
        else {
            varList = varList.filter(function (variable) { return variable.trainable; });
        }
        return this.engine.variableGradientsAndValue(f, varList);
    };
    NDArrayMath.prototype.valueAndGradients = function (f, x) {
        var keys = x instanceof ndarray_1.NDArray ? null : Object.keys(x);
        var xs = util.flattenNameArrayMap(x, keys);
        var returnValue = true;
        var valueAndGradients = this.engine.gradients(f, xs, returnValue);
        var gradients;
        if (x instanceof ndarray_1.NDArray) {
            gradients = valueAndGradients.gradients[0];
        }
        else {
            gradients =
                util.unflattenToNameArrayMap(keys, valueAndGradients.gradients);
        }
        return { value: valueAndGradients.value, gradients: gradients };
    };
    NDArrayMath.prototype.customGradient = function (name, f, inputs) {
        return this.engine.customGradient(f, inputs, name == null ? '' : name);
    };
    NDArrayMath.prototype.disposeData = function (dataId) {
        if (!this.registeredArrays.has(dataId)) {
            return;
        }
        var refCount = this.registeredArrays.get(dataId);
        if (refCount <= 1) {
            this.registeredArrays.delete(dataId);
            this.backend.disposeData(dataId);
        }
        else {
            this.registeredArrays.set(dataId, refCount - 1);
        }
    };
    return NDArrayMath;
}());
exports.NDArrayMath = NDArrayMath;
//# sourceMappingURL=math.js.map