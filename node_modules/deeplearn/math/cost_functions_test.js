"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var cost_functions_1 = require("./cost_functions");
var ndarray_1 = require("./ndarray");
var tests = function (it) {
    it('Square cost', function (math) {
        var y = ndarray_1.Array1D.new([1, 3, -2]);
        var target = ndarray_1.Array1D.new([0, 3, -1.5]);
        var square = new cost_functions_1.SquareCostFunc();
        var cost = square.cost(math, y, target);
        test_util.expectNumbersClose(cost.get(0), 1 / 2);
        test_util.expectNumbersClose(cost.get(1), 0 / 2);
        test_util.expectNumbersClose(cost.get(2), 0.25 / 2);
    });
    it('Square derivative', function (math) {
        var y = ndarray_1.Array1D.new([1, 3, -2]);
        var target = ndarray_1.Array1D.new([0, 3, -1.5]);
        var square = new cost_functions_1.SquareCostFunc();
        var dy = square.der(math, y, target);
        test_util.expectNumbersClose(dy.get(0), 1);
        test_util.expectNumbersClose(dy.get(1), 0);
        test_util.expectNumbersClose(dy.get(2), -0.5);
    });
};
test_util.describeMathCPU('Square cost', [tests]);
test_util.describeMathGPU('Square cost', [tests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=cost_functions_test.js.map