"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var ndarray_1 = require("../math/ndarray");
var test_util = require("../test_util");
var util = require("../util");
var activation_functions_1 = require("./activation_functions");
describe('Activation functions', function () {
    var math = environment_1.ENV.math;
    it('Tanh output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var tanH = new activation_functions_1.TanHFunc();
        var y = tanH.output(math, x);
        test_util.expectNumbersClose(y.get(0), util.tanh(x.get(0)));
        test_util.expectNumbersClose(y.get(1), util.tanh(x.get(1)));
        test_util.expectNumbersClose(y.get(2), util.tanh(x.get(2)));
        test_util.expectNumbersClose(y.get(3), 1);
        test_util.expectNumbersClose(y.get(4), -1);
        test_util.expectNumbersClose(y.get(5), 0);
    });
    it('Tanh derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var tanH = new activation_functions_1.TanHFunc();
        var y = tanH.output(math, x);
        var dx = tanH.der(math, x, y);
        test_util.expectNumbersClose(dx.get(0), 1 - Math.pow(y.get(0), 2));
        test_util.expectNumbersClose(dx.get(1), 1 - Math.pow(y.get(1), 2));
        test_util.expectNumbersClose(dx.get(2), 1 - Math.pow(y.get(2), 2));
    });
    it('ReLU output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2]);
        var relu = new activation_functions_1.ReLUFunc();
        var y = relu.output(math, x);
        test_util.expectNumbersClose(y.get(0), 1);
        test_util.expectNumbersClose(y.get(1), 3);
        test_util.expectNumbersClose(y.get(2), 0);
    });
    it('ReLU derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2]);
        var relu = new activation_functions_1.ReLUFunc();
        var y = relu.output(math, x);
        var dx = relu.der(math, x, y);
        test_util.expectNumbersClose(dx.get(0), 1);
        test_util.expectNumbersClose(dx.get(1), 1);
        test_util.expectNumbersClose(dx.get(2), 0);
    });
    it('LeakyRelu output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2]);
        var relu = new activation_functions_1.LeakyReluFunc(0.2);
        var y = relu.output(math, x);
        test_util.expectNumbersClose(y.get(0), 1);
        test_util.expectNumbersClose(y.get(1), 3);
        test_util.expectNumbersClose(y.get(2), -0.4);
    });
    it('LeakyRelu derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2]);
        var relu = new activation_functions_1.LeakyReluFunc(0.2);
        var y = relu.output(math, x);
        var dx = relu.der(math, x, y);
        test_util.expectNumbersClose(dx.get(0), 1);
        test_util.expectNumbersClose(dx.get(1), 1);
        test_util.expectNumbersClose(dx.get(2), 0.2);
    });
    it('Sigmoid output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var sigmoid = new activation_functions_1.SigmoidFunc();
        var y = sigmoid.output(math, x);
        test_util.expectNumbersClose(y.get(0), 1 / (1 + Math.exp(-1)));
        test_util.expectNumbersClose(y.get(1), 1 / (1 + Math.exp(-3)));
        test_util.expectNumbersClose(y.get(2), 1 / (1 + Math.exp(2)));
        test_util.expectNumbersClose(y.get(3), 1);
        test_util.expectNumbersClose(y.get(4), 0);
        test_util.expectNumbersClose(y.get(5), 0.5);
    });
    it('Sigmoid derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var sigmoid = new activation_functions_1.SigmoidFunc();
        var y = sigmoid.output(math, x);
        var dx = sigmoid.der(math, x, y);
        test_util.expectNumbersClose(dx.get(0), y.get(0) * (1 - y.get(0)));
        test_util.expectNumbersClose(dx.get(1), y.get(1) * (1 - y.get(1)));
        test_util.expectNumbersClose(dx.get(2), y.get(2) * (1 - y.get(2)));
    });
});
//# sourceMappingURL=activation_functions_test.js.map