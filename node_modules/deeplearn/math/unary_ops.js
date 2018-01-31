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
var Ops = (function () {
    function Ops() {
    }
    Ops.neg = function (x) {
        return environment_1.ENV.engine.executeKernel('Neg', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.neg(); } };
        });
    };
    Ops.ceil = function (x) {
        return environment_1.ENV.engine.executeKernel('Ceil', { inputs: { x: x } });
    };
    Ops.floor = function (x) {
        return environment_1.ENV.engine.executeKernel('Floor', { inputs: { x: x } });
    };
    Ops.exp = function (x) {
        return environment_1.ENV.engine.executeKernel('Exp', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.mul(y); } };
        });
    };
    Ops.log = function (x) {
        return environment_1.ENV.engine.executeKernel('Log', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.div(x.toFloat()); } };
        });
    };
    Ops.sqrt = function (x) {
        return environment_1.ENV.engine.executeKernel('Sqrt', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.div(x.toFloat().sqrt().mul(ndarray_1.Scalar.new(2))); } };
        });
    };
    Ops.square = function (x) {
        return environment_1.ENV.engine.executeKernel('Square', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.mul(x.toFloat().mul(ndarray_1.Scalar.new(2))); } };
        });
    };
    Ops.abs = function (x) {
        return environment_1.ENV.engine.executeKernel('Abs', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.mul(x.toFloat().step(-1)); } };
        });
    };
    Ops.clip = function (x, min, max) {
        util.assert((min <= max), "Error in clip: min (" + min + ") must be" +
            ("less than or equal to max (" + max + ")."));
        return environment_1.ENV.engine.executeKernel('Clip', { inputs: { x: x }, args: { min: min, max: max } });
    };
    Ops.relu = function (x) {
        return environment_1.ENV.engine.executeKernel('Relu', { inputs: { x: x } }, function (dy, y) {
            var stepRes = x.step();
            return { x: function () { return dy.mul(stepRes.toFloat()); } };
        });
    };
    Ops.elu = function (x) {
        var der = function (dy) {
            return {
                x: function () { return dy.mul(eluDer(x)); },
                alpha: function () {
                    throw new Error('Derivative of prelu with respect to alpha is ' +
                        'not implemented yet');
                }
            };
        };
        return environment_1.ENV.engine.executeKernel('Elu', { inputs: { x: x } }, der);
    };
    Ops.selu = function (x) {
        return environment_1.ENV.engine.executeKernel('Selu', { inputs: { x: x } });
    };
    Ops.leakyRelu = function (x, alpha) {
        if (alpha === void 0) { alpha = 0.2; }
        return environment_1.ENV.engine.executeKernel('LeakyRelu', { inputs: { x: x }, args: { alpha: alpha } });
    };
    Ops.prelu = function (x, alpha) {
        var der = function (dy) {
            return {
                x: function () { return dy.mul(preluDer(x, alpha)); },
                alpha: function () {
                    throw new Error('Derivative of prelu with respect to alpha is ' +
                        'not implemented yet');
                }
            };
        };
        return environment_1.ENV.engine.executeKernel('PReLU', { inputs: { x: x, alpha: alpha } }, der);
    };
    Ops.sigmoid = function (x) {
        return environment_1.ENV.engine.executeKernel('Sigmoid', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.mul(y.mul(ndarray_1.Scalar.new(1).sub(y))); } };
        });
    };
    Ops.sin = function (x) {
        return environment_1.ENV.engine.executeKernel('Sin', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return x.toFloat().cos().mul(dy); } };
        });
    };
    Ops.cos = function (x) {
        return environment_1.ENV.engine.executeKernel('Cos', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return x.toFloat().sin().neg().mul(dy); } };
        });
    };
    Ops.tan = function (x) {
        return environment_1.ENV.engine.executeKernel('Tan', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.div(x.cos().square()); } };
        });
    };
    Ops.asin = function (x) {
        return environment_1.ENV.engine.executeKernel('Asin', { inputs: { x: x } }, function (dy, y) {
            return {
                x: function () { return dy.div(Ops.sqrt(ndarray_1.Scalar.new(1).sub(x.toFloat().square()))); }
            };
        });
    };
    Ops.acos = function (x) {
        return environment_1.ENV.engine.executeKernel('Acos', { inputs: { x: x } }, function (dy, y) {
            return {
                x: function () { return dy.div(Ops.sqrt(ndarray_1.Scalar.new(1).sub(x.toFloat().square()))).neg(); }
            };
        });
    };
    Ops.atan = function (x) {
        return environment_1.ENV.engine.executeKernel('Atan', { inputs: { x: x } }, function (dy, y) {
            return { x: function () { return dy.div(ndarray_1.Scalar.new(1).add(x.toFloat().square())); } };
        });
    };
    Ops.sinh = function (x) {
        return environment_1.ENV.engine.executeKernel('Sinh', { inputs: { x: x } });
    };
    Ops.cosh = function (x) {
        return environment_1.ENV.engine.executeKernel('Cosh', { inputs: { x: x } });
    };
    Ops.tanh = function (x) {
        return environment_1.ENV.engine.executeKernel('Tanh', { inputs: { x: x } });
    };
    Ops.step = function (x, alpha) {
        if (alpha === void 0) { alpha = 0.0; }
        return environment_1.ENV.engine.executeKernel('Step', { inputs: { x: x }, args: { alpha: alpha } });
    };
    __decorate([
        decorators_1.operation
    ], Ops, "neg", null);
    __decorate([
        decorators_1.operation
    ], Ops, "ceil", null);
    __decorate([
        decorators_1.operation
    ], Ops, "floor", null);
    __decorate([
        decorators_1.operation
    ], Ops, "exp", null);
    __decorate([
        decorators_1.operation
    ], Ops, "log", null);
    __decorate([
        decorators_1.operation
    ], Ops, "sqrt", null);
    __decorate([
        decorators_1.operation
    ], Ops, "square", null);
    __decorate([
        decorators_1.operation
    ], Ops, "abs", null);
    __decorate([
        decorators_1.operation
    ], Ops, "clip", null);
    __decorate([
        decorators_1.operation
    ], Ops, "relu", null);
    __decorate([
        decorators_1.operation
    ], Ops, "elu", null);
    __decorate([
        decorators_1.operation
    ], Ops, "selu", null);
    __decorate([
        decorators_1.operation
    ], Ops, "leakyRelu", null);
    __decorate([
        decorators_1.operation
    ], Ops, "prelu", null);
    __decorate([
        decorators_1.operation
    ], Ops, "sigmoid", null);
    __decorate([
        decorators_1.operation
    ], Ops, "sin", null);
    __decorate([
        decorators_1.operation
    ], Ops, "cos", null);
    __decorate([
        decorators_1.operation
    ], Ops, "tan", null);
    __decorate([
        decorators_1.operation
    ], Ops, "asin", null);
    __decorate([
        decorators_1.operation
    ], Ops, "acos", null);
    __decorate([
        decorators_1.operation
    ], Ops, "atan", null);
    __decorate([
        decorators_1.operation
    ], Ops, "sinh", null);
    __decorate([
        decorators_1.operation
    ], Ops, "cosh", null);
    __decorate([
        decorators_1.operation
    ], Ops, "tanh", null);
    __decorate([
        decorators_1.operation
    ], Ops, "step", null);
    return Ops;
}());
exports.Ops = Ops;
function preluDer(x, alpha) {
    return environment_1.ENV.engine.executeKernel('PReLUDer', { inputs: { x: x, alpha: alpha } });
}
function eluDer(x) {
    return environment_1.ENV.engine.executeKernel('EluDer', { inputs: { x: x } });
}
//# sourceMappingURL=unary_ops.js.map