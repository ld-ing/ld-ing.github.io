"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var tests = function (it) {
    it('simple assign', function (math) {
        var v = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]));
        test_util.expectArraysClose(v, [1, 2, 3]);
        v.assign(ndarray_1.Array1D.new([4, 5, 6]));
        test_util.expectArraysClose(v, [4, 5, 6]);
    });
    it('default names are unique', function (math) {
        var v = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]));
        expect(v.name).not.toBeNull();
        var v2 = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]));
        expect(v2.name).not.toBeNull();
        expect(v.name).not.toBe(v2.name);
    });
    it('user provided name', function (math) {
        var v = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]), true, 'myName');
        expect(v.name).toBe('myName');
    });
    it('if name already used, throw error', function (math) {
        ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]), true, 'myName');
        expect(function () { return ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]), true, 'myName'); })
            .toThrowError();
    });
    it('math ops can take variables', function (math) {
        var value = ndarray_1.Array1D.new([1, 2, 3]);
        var v = ndarray_1.variable(value);
        var res = math.sum(v);
        test_util.expectArraysClose(res, [6]);
    });
    it('variables are not affected by scopes', function (math) {
        var v;
        expect(math.getNumArrays()).toBe(0);
        math.scope(function () {
            var value = ndarray_1.Array1D.new([1, 2, 3], 'float32');
            expect(math.getNumArrays()).toBe(1);
            v = ndarray_1.variable(value);
            expect(math.getNumArrays()).toBe(1);
        });
        expect(math.getNumArrays()).toBe(1);
        test_util.expectArraysClose(v, [1, 2, 3]);
        v.dispose();
        expect(math.getNumArrays()).toBe(0);
    });
    it('variables are assignable to ndarrays', function () {
        var x0 = null;
        var y0 = x0;
        expect(y0).toBeNull();
        var x1 = null;
        var y1 = x1;
        expect(y1).toBeNull();
        var x2 = null;
        var y2 = x2;
        expect(y2).toBeNull();
        var x3 = null;
        var y3 = x3;
        expect(y3).toBeNull();
        var x4 = null;
        var y4 = x4;
        expect(y4).toBeNull();
        var xh = null;
        var yh = xh;
        expect(yh).toBeNull();
    });
    it('assign will dispose old data', function (math) {
        var v;
        v = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]));
        expect(math.getNumArrays()).toBe(1);
        test_util.expectArraysClose(v, [1, 2, 3]);
        var secondArray = ndarray_1.Array1D.new([4, 5, 6]);
        expect(math.getNumArrays()).toBe(2);
        v.assign(secondArray);
        test_util.expectArraysClose(v, [4, 5, 6]);
        expect(math.getNumArrays()).toBe(1);
        v.dispose();
        expect(math.getNumArrays()).toBe(0);
    });
    it('shape must match', function (math) {
        var v = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]));
        expect(function () { return v.assign(ndarray_1.Array1D.new([1, 2])); }).toThrowError();
        expect(function () { return v.assign(ndarray_1.Array2D.new([1, 2], [3, 4])); }).toThrowError();
    });
    it('dtype must match', function (math) {
        var v = ndarray_1.variable(ndarray_1.Array1D.new([1, 2, 3]));
        expect(function () { return v.assign(ndarray_1.Array1D.new([1, 1, 1], 'int32')); })
            .toThrowError();
        expect(function () { return v.assign(ndarray_1.Array1D.new([true, false, true], 'bool')); })
            .toThrowError();
    });
};
test_util.describeMathCPU('Variables', [tests]);
test_util.describeMathGPU('Variables', [tests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=variable_test.js.map