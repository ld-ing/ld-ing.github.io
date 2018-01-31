"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var tests = function (it) {
    it('Depth 1 throws error', function (math) {
        var indices = ndarray_1.Array1D.new([0, 0, 0]);
        expect(function () { return math.oneHot(indices, 1); }).toThrowError();
    });
    it('Depth 2, diagonal', function (math) {
        var indices = ndarray_1.Array1D.new([0, 1]);
        var res = math.oneHot(indices, 2);
        expect(res.shape).toEqual([2, 2]);
        test_util.expectArraysClose(res, [1, 0, 0, 1]);
    });
    it('Depth 2, transposed diagonal', function (math) {
        var indices = ndarray_1.Array1D.new([1, 0]);
        var res = math.oneHot(indices, 2);
        expect(res.shape).toEqual([2, 2]);
        test_util.expectArraysClose(res, [0, 1, 1, 0]);
    });
    it('Depth 3, 4 events', function (math) {
        var indices = ndarray_1.Array1D.new([2, 1, 2, 0]);
        var res = math.oneHot(indices, 3);
        expect(res.shape).toEqual([4, 3]);
        test_util.expectArraysClose(res, [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
    });
    it('Depth 2 onValue=3, offValue=-2', function (math) {
        var indices = ndarray_1.Array1D.new([0, 1]);
        var res = math.oneHot(indices, 2, 3, -2);
        expect(res.shape).toEqual([2, 2]);
        test_util.expectArraysClose(res, [3, -2, -2, 3]);
    });
};
test_util.describeMathCPU('oneHot', [tests]);
test_util.describeMathGPU('oneHot', [tests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=onehot_test.js.map