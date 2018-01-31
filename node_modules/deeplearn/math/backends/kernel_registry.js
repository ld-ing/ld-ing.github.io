"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var ndarray_1 = require("../ndarray");
function executeKernel(backend, kernelName, inputAndArgs) {
    if (kernelName === 'MatMul') {
        var config = inputAndArgs;
        return backend.matMul(config.inputs.a, config.inputs.b, config.args.aOrientation, config.args.bOrientation);
    }
    else if (kernelName === 'Clone') {
        var config = inputAndArgs;
        return backend.clone(config.inputs.x);
    }
    else if (kernelName === 'Slice1D') {
        var config = inputAndArgs;
        return backend.slice1D(config.inputs.x, config.args.begin, config.args.size);
    }
    else if (kernelName === 'Slice2D') {
        var config = inputAndArgs;
        return backend.slice2D(config.inputs.x, config.args.begin, config.args.size);
    }
    else if (kernelName === 'Slice3D') {
        var config = inputAndArgs;
        return backend.slice3D(config.inputs.x, config.args.begin, config.args.size);
    }
    else if (kernelName === 'Slice4D') {
        var config = inputAndArgs;
        return backend.slice4D(config.inputs.x, config.args.begin, config.args.size);
    }
    else if (kernelName === 'Reverse4D') {
        var config = inputAndArgs;
        return backend.reverse4D(config.inputs.x, config.args.axis);
    }
    else if (kernelName === 'Concat') {
        var config = inputAndArgs;
        return backend.concat(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Neg') {
        var config = inputAndArgs;
        return backend.neg(config.inputs.x);
    }
    else if (kernelName === 'Add') {
        var config = inputAndArgs;
        return backend.add(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Sub') {
        var config = inputAndArgs;
        return backend.subtract(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Mul') {
        var config = inputAndArgs;
        return backend.multiply(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Div') {
        var config = inputAndArgs;
        return backend.divide(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Sum') {
        var config = inputAndArgs;
        return backend.sum(config.inputs.x, config.args.axes);
    }
    else if (kernelName === 'ArgMax') {
        var config = inputAndArgs;
        return backend.argMax(config.inputs.x, config.args.axes);
    }
    else if (kernelName === 'ArgMin') {
        var config = inputAndArgs;
        return backend.argMin(config.inputs.x, config.args.axes);
    }
    else if (kernelName === 'Equal') {
        var config = inputAndArgs;
        return backend.equal(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'NotEqual') {
        var config = inputAndArgs;
        return backend.notEqual(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Less') {
        var config = inputAndArgs;
        return backend.less(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'LessEqual') {
        var config = inputAndArgs;
        return backend.lessEqual(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Greater') {
        var config = inputAndArgs;
        return backend.greater(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'GreaterEqual') {
        var config = inputAndArgs;
        return backend.greaterEqual(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'LogicalAnd') {
        var config = inputAndArgs;
        return backend.logicalAnd(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'LogicalOr') {
        var config = inputAndArgs;
        return backend.logicalOr(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Where') {
        var config = inputAndArgs;
        return backend.where(config.inputs.condition, config.inputs.a, config.inputs.b, config.args.dtype);
    }
    else if (kernelName === 'TopKValues') {
        var config = inputAndArgs;
        return backend.topKValues(config.inputs.x, config.args.k);
    }
    else if (kernelName === 'TopKIndices') {
        var config = inputAndArgs;
        return backend.topKIndices(config.inputs.x, config.args.k);
    }
    else if (kernelName === 'Min') {
        var config = inputAndArgs;
        return backend.min(config.inputs.x, config.args.axes);
    }
    else if (kernelName === 'Minimum') {
        var config = inputAndArgs;
        return backend.minimum(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Max') {
        var config = inputAndArgs;
        return backend.max(config.inputs.x, config.args.axes);
    }
    else if (kernelName === 'Maximum') {
        var config = inputAndArgs;
        return backend.maximum(config.inputs.a, config.inputs.b);
    }
    else if (kernelName === 'Ceil') {
        var config = inputAndArgs;
        return backend.ceil(config.inputs.x);
    }
    else if (kernelName === 'Floor') {
        var config = inputAndArgs;
        return backend.floor(config.inputs.x);
    }
    else if (kernelName === 'Pow') {
        var config = inputAndArgs;
        return backend.pow(config.inputs.base, config.inputs.exp);
    }
    else if (kernelName === 'Exp') {
        var config = inputAndArgs;
        return backend.exp(config.inputs.x);
    }
    else if (kernelName === 'Log') {
        var config = inputAndArgs;
        return backend.log(config.inputs.x);
    }
    else if (kernelName === 'Sqrt') {
        var config = inputAndArgs;
        return backend.sqrt(config.inputs.x);
    }
    else if (kernelName === 'Square') {
        var config = inputAndArgs;
        return backend.square(config.inputs.x);
    }
    else if (kernelName === 'Relu') {
        var config = inputAndArgs;
        return backend.relu(config.inputs.x);
    }
    else if (kernelName === 'Reshape') {
        var config = inputAndArgs;
        var x = config.inputs.x;
        var newShape = config.args.newShape;
        return ndarray_1.NDArray.make(newShape, { dataId: x.dataId }, x.dtype);
    }
    else if (kernelName === 'Cast') {
        var config = inputAndArgs;
        var x = config.inputs.x;
        var newDType = config.args.newDType;
        if (!util.hasEncodingLoss(x.dtype, newDType)) {
            return ndarray_1.NDArray.make(x.shape, { dataId: x.dataId }, newDType);
        }
        if (newDType === 'int32') {
            return backend.int(x);
        }
        else if (newDType === 'bool') {
            return backend.notEqual(x, ndarray_1.Scalar.new(0, x.dtype));
        }
        else {
            throw new Error("Error in Cast: unknown dtype argument (" + newDType + ")");
        }
    }
    else if (kernelName === 'LeakyRelu') {
        var config = inputAndArgs;
        return backend.leakyRelu(config.inputs.x, config.args.alpha);
    }
    else if (kernelName === 'PReLU') {
        var config = inputAndArgs;
        return backend.prelu(config.inputs.x, config.inputs.alpha);
    }
    else if (kernelName === 'PReLUDer') {
        var config = inputAndArgs;
        return backend.preluDer(config.inputs.x, config.inputs.alpha);
    }
    else if (kernelName === 'Elu') {
        var config = inputAndArgs;
        return backend.elu(config.inputs.x);
    }
    else if (kernelName === 'EluDer') {
        var config = inputAndArgs;
        return backend.eluDer(config.inputs.x);
    }
    else if (kernelName === 'Selu') {
        var config = inputAndArgs;
        return backend.selu(config.inputs.x);
    }
    else if (kernelName === 'Abs') {
        var config = inputAndArgs;
        return backend.abs(config.inputs.x);
    }
    else if (kernelName === 'Sigmoid') {
        var config = inputAndArgs;
        return backend.sigmoid(config.inputs.x);
    }
    else if (kernelName === 'Step') {
        var config = inputAndArgs;
        return backend.step(config.inputs.x, config.args.alpha);
    }
    else if (kernelName === 'Sin') {
        var config = inputAndArgs;
        return backend.sin(config.inputs.x);
    }
    else if (kernelName === 'Cos') {
        var config = inputAndArgs;
        return backend.cos(config.inputs.x);
    }
    else if (kernelName === 'Tan') {
        var config = inputAndArgs;
        return backend.tan(config.inputs.x);
    }
    else if (kernelName === 'Asin') {
        var config = inputAndArgs;
        return backend.asin(config.inputs.x);
    }
    else if (kernelName === 'Acos') {
        var config = inputAndArgs;
        return backend.acos(config.inputs.x);
    }
    else if (kernelName === 'Atan') {
        var config = inputAndArgs;
        return backend.atan(config.inputs.x);
    }
    else if (kernelName === 'Sinh') {
        var config = inputAndArgs;
        return backend.sinh(config.inputs.x);
    }
    else if (kernelName === 'Cosh') {
        var config = inputAndArgs;
        return backend.cosh(config.inputs.x);
    }
    else if (kernelName === 'Tanh') {
        var config = inputAndArgs;
        return backend.tanh(config.inputs.x);
    }
    else if (kernelName === 'Clip') {
        var config = inputAndArgs;
        return backend.clip(config.inputs.x, config.args.min, config.args.max);
    }
    else if (kernelName === 'Tile') {
        var config = inputAndArgs;
        return backend.tile(config.inputs.x, config.args.reps);
    }
    else if (kernelName === 'Gather') {
        var config = inputAndArgs;
        return backend.gather(config.inputs.x, config.inputs.indices, config.args.axis);
    }
    else if (kernelName === 'Pad1D') {
        var config = inputAndArgs;
        return backend.pad1D(config.inputs.x, config.args.paddings, config.args.constantValue);
    }
    else if (kernelName === 'Pad2D') {
        var config = inputAndArgs;
        return backend.pad2D(config.inputs.x, config.args.paddings, config.args.constantValue);
    }
    else if (kernelName === 'Transpose') {
        var config = inputAndArgs;
        return backend.transpose(config.inputs.x, config.args.perm);
    }
    else if (kernelName === 'Conv2D') {
        var config = inputAndArgs;
        return backend.conv2d(config.inputs.x, config.inputs.filter, config.inputs.bias, config.args.convInfo);
    }
    else if (kernelName === 'Conv2DDerInput') {
        var config = inputAndArgs;
        return backend.conv2dDerInput(config.inputs.dy, config.inputs.filter, config.args.convInfo);
    }
    else if (kernelName === 'Conv2DDerFilter') {
        var config = inputAndArgs;
        return backend.conv2dDerFilter(config.inputs.x, config.inputs.dy, config.args.convInfo);
    }
    else if (kernelName === 'Conv2DDerBias') {
        var config = inputAndArgs;
        return backend.conv2dDerBias(config.inputs.dy);
    }
    else if (kernelName === 'DepthwiseConv2D') {
        var config = inputAndArgs;
        return backend.depthwiseConv2D(config.inputs.x, config.inputs.filter, config.args.convInfo);
    }
    else if (kernelName === 'MaxPool') {
        var config = inputAndArgs;
        return backend.maxPool(config.inputs.x, config.args.convInfo);
    }
    else if (kernelName === 'MaxPoolBackprop') {
        var config = inputAndArgs;
        return backend.maxPoolBackprop(config.inputs.dy, config.inputs.x, config.args.convInfo);
    }
    else if (kernelName === 'AvgPool') {
        var config = inputAndArgs;
        return backend.avgPool(config.inputs.x, config.args.convInfo);
    }
    else if (kernelName === 'AvgPoolBackprop') {
        var config = inputAndArgs;
        return backend.avgPoolBackprop(config.inputs.dy, config.inputs.x, config.args.convInfo);
    }
    else if (kernelName === 'MinPool') {
        var config = inputAndArgs;
        return backend.minPool(config.inputs.x, config.args.convInfo);
    }
    else if (kernelName === 'ResizeBilinear') {
        var config = inputAndArgs;
        return backend.resizeBilinear(config.inputs.x, config.args.newHeight, config.args.newWidth, config.args.alignCorners);
    }
    else if (kernelName === 'BatchNorm4D') {
        var config = inputAndArgs;
        return backend.batchNormalization4D(config.inputs.x, config.inputs.mean, config.inputs.variance, config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
    }
    else if (kernelName === 'BatchNorm3D') {
        var config = inputAndArgs;
        return backend.batchNormalization3D(config.inputs.x, config.inputs.mean, config.inputs.variance, config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
    }
    else if (kernelName === 'BatchNorm2D') {
        var config = inputAndArgs;
        return backend.batchNormalization2D(config.inputs.x, config.inputs.mean, config.inputs.variance, config.args.varianceEpsilon, config.inputs.scale, config.inputs.offset);
    }
    else if (kernelName === 'LRN4D') {
        var config = inputAndArgs;
        return backend.localResponseNormalization4D(config.inputs.x, config.args.radius, config.args.bias, config.args.alpha, config.args.beta, config.args.normRegion);
    }
    else if (kernelName === 'Multinomial') {
        var config = inputAndArgs;
        return backend.multinomial(config.inputs.probs, config.args.numSamples, config.args.seed);
    }
    else if (kernelName === 'OneHot') {
        var config = inputAndArgs;
        return backend.oneHot(config.inputs.indices, config.args.depth, config.args.onValue, config.args.offValue);
    }
    throw new Error("No backend method found for kernel " + kernelName);
}
exports.executeKernel = executeKernel;
//# sourceMappingURL=kernel_registry.js.map