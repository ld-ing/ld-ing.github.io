"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2]);
            var min = -1;
            var max = 50;
            var result = math.clip(a, min, max);
            test_util.expectArraysClose(result, [3, -1, 0, 50, -1, 2]);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2, NaN]);
            var min = -1;
            var max = 50;
            var result = math.clip(a, min, max);
            test_util.expectArraysClose(result, [3, -1, 0, 50, -1, 2, NaN]);
        });
        it('min greater than max', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2]);
            var min = 1;
            var max = -1;
            var f = function () {
                math.clip(a, min, max);
            };
            expect(f).toThrowError();
        });
    };
    test_util.describeMathCPU('clip', [tests]);
    test_util.describeMathGPU('clip', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=clip_test.js.map