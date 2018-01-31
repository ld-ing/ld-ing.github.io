"use strict";
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
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var types_1 = require("./types");
var tests = function (it) {
    it('NDArrays of arbitrary size', function () {
        var t = ndarray_1.Array1D.new([1, 2, 3]);
        expect(t instanceof ndarray_1.Array1D).toBe(true);
        expect(t.rank).toBe(1);
        expect(t.size).toBe(3);
        test_util.expectArraysClose(t, [1, 2, 3]);
        expect(t.get(4)).toBeUndefined();
        t = ndarray_1.Array2D.new([1, 3], [1, 2, 3]);
        expect(t instanceof ndarray_1.Array2D).toBe(true);
        expect(t.rank).toBe(2);
        expect(t.size).toBe(3);
        test_util.expectArraysClose(t, [1, 2, 3]);
        expect(t.get(4)).toBeUndefined();
        t = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        expect(t instanceof ndarray_1.Array2D).toBe(true);
        expect(t.rank).toBe(2);
        expect(t.size).toBe(6);
        test_util.expectArraysClose(t, [1, 2, 3, 4, 5, 6]);
        expect(t.get(5, 3)).toBeUndefined();
        expect(function () { return ndarray_1.Array2D.new([1, 2], [1]); }).toThrowError();
    });
    it('NDArrays of explicit size', function () {
        var t = ndarray_1.Array1D.new([5, 3, 2]);
        expect(t.rank).toBe(1);
        expect(t.shape).toEqual([3]);
        test_util.expectNumbersClose(t.get(1), 3);
        expect(function () { return ndarray_1.Array3D.new([1, 2, 3, 5], [1, 2]); }).toThrowError();
        var t4 = ndarray_1.Array4D.new([1, 2, 1, 2], [1, 2, 3, 4]);
        test_util.expectNumbersClose(t4.get(0, 0, 0, 0), 1);
        test_util.expectNumbersClose(t4.get(0, 0, 0, 1), 2);
        test_util.expectNumbersClose(t4.get(0, 1, 0, 0), 3);
        test_util.expectNumbersClose(t4.get(0, 1, 0, 1), 4);
        var t4Like = dl.clone(t4);
        t4.set(10, 0, 0, 0, 1);
        test_util.expectNumbersClose(t4.get(0, 0, 0, 1), 10);
        test_util.expectNumbersClose(t4Like.get(0, 0, 0, 1), 2);
        var x = dl.ones([3, 4, 2]);
        expect(x.rank).toBe(3);
        expect(x.size).toBe(24);
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 4; j++) {
                for (var k = 0; k < 2; k++) {
                    test_util.expectNumbersClose(x.get(i, j, k), 1);
                }
            }
        }
        var z = dl.zeros([3, 4, 2]);
        expect(z.rank).toBe(3);
        expect(z.size).toBe(24);
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 4; j++) {
                for (var k = 0; k < 2; k++) {
                    test_util.expectNumbersClose(z.get(i, j, k), 0);
                }
            }
        }
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([3, 2, 1]);
        test_util.expectNumbersClose(a.get(1, 2), 6);
        b.set(10, 2, 1, 0);
        test_util.expectNumbersClose(a.get(1, 2), 10);
    });
    it('NDArray dataSync CPU --> GPU', function () {
        var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
        test_util.expectArraysClose(a.dataSync(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('NDArray.data() CPU --> GPU', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, _a, _b;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
                    _b = (_a = test_util).expectArraysClose;
                    return [4, a.data()];
                case 1:
                    _b.apply(_a, [_c.sent(), new Float32Array([1, 2, 3, 4, 5, 6])]);
                    return [2];
            }
        });
    }); });
    it('Scalar basic methods', function () {
        var a = ndarray_1.Scalar.new(5);
        test_util.expectNumbersClose(a.get(), 5);
        test_util.expectArraysClose(a, [5]);
        expect(a.rank).toBe(0);
        expect(a.size).toBe(1);
        expect(a.shape).toEqual([]);
    });
    it('indexToLoc Scalar', function () {
        var a = ndarray_1.Scalar.new(0);
        expect(a.indexToLoc(0)).toEqual([]);
        var b = dl.zeros([]);
        expect(b.indexToLoc(0)).toEqual([]);
    });
    it('indexToLoc Array1D', function () {
        var a = dl.zeros([3]);
        expect(a.indexToLoc(0)).toEqual([0]);
        expect(a.indexToLoc(1)).toEqual([1]);
        expect(a.indexToLoc(2)).toEqual([2]);
        var b = dl.zeros([3]);
        expect(b.indexToLoc(0)).toEqual([0]);
        expect(b.indexToLoc(1)).toEqual([1]);
        expect(b.indexToLoc(2)).toEqual([2]);
    });
    it('indexToLoc Array2D', function () {
        var a = dl.zeros([3, 2]);
        expect(a.indexToLoc(0)).toEqual([0, 0]);
        expect(a.indexToLoc(1)).toEqual([0, 1]);
        expect(a.indexToLoc(2)).toEqual([1, 0]);
        expect(a.indexToLoc(3)).toEqual([1, 1]);
        expect(a.indexToLoc(4)).toEqual([2, 0]);
        expect(a.indexToLoc(5)).toEqual([2, 1]);
        var b = dl.zeros([3, 2]);
        expect(b.indexToLoc(0)).toEqual([0, 0]);
        expect(b.indexToLoc(1)).toEqual([0, 1]);
        expect(b.indexToLoc(2)).toEqual([1, 0]);
        expect(b.indexToLoc(3)).toEqual([1, 1]);
        expect(b.indexToLoc(4)).toEqual([2, 0]);
        expect(b.indexToLoc(5)).toEqual([2, 1]);
    });
    it('indexToLoc Array3D', function () {
        var a = dl.zeros([3, 2, 2]);
        expect(a.indexToLoc(0)).toEqual([0, 0, 0]);
        expect(a.indexToLoc(1)).toEqual([0, 0, 1]);
        expect(a.indexToLoc(2)).toEqual([0, 1, 0]);
        expect(a.indexToLoc(3)).toEqual([0, 1, 1]);
        expect(a.indexToLoc(4)).toEqual([1, 0, 0]);
        expect(a.indexToLoc(5)).toEqual([1, 0, 1]);
        expect(a.indexToLoc(11)).toEqual([2, 1, 1]);
        var b = dl.zeros([3, 2, 2]);
        expect(b.indexToLoc(0)).toEqual([0, 0, 0]);
        expect(b.indexToLoc(1)).toEqual([0, 0, 1]);
        expect(b.indexToLoc(2)).toEqual([0, 1, 0]);
        expect(b.indexToLoc(3)).toEqual([0, 1, 1]);
        expect(b.indexToLoc(4)).toEqual([1, 0, 0]);
        expect(b.indexToLoc(5)).toEqual([1, 0, 1]);
        expect(b.indexToLoc(11)).toEqual([2, 1, 1]);
    });
    it('indexToLoc NDArray 5D', function () {
        var values = new Float32Array([1, 2, 3, 4]);
        var a = ndarray_1.NDArray.make([2, 1, 1, 1, 2], { values: values });
        expect(a.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
        expect(a.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
        expect(a.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
        expect(a.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
    });
    it('locToIndex Scalar', function () {
        var a = ndarray_1.Scalar.new(0);
        expect(a.locToIndex([])).toEqual(0);
        var b = dl.zeros([]);
        expect(b.locToIndex([])).toEqual(0);
    });
    it('locToIndex Array1D', function () {
        var a = dl.zeros([3]);
        expect(a.locToIndex([0])).toEqual(0);
        expect(a.locToIndex([1])).toEqual(1);
        expect(a.locToIndex([2])).toEqual(2);
        var b = dl.zeros([3]);
        expect(b.locToIndex([0])).toEqual(0);
        expect(b.locToIndex([1])).toEqual(1);
        expect(b.locToIndex([2])).toEqual(2);
    });
    it('locToIndex Array2D', function () {
        var a = dl.zeros([3, 2]);
        expect(a.locToIndex([0, 0])).toEqual(0);
        expect(a.locToIndex([0, 1])).toEqual(1);
        expect(a.locToIndex([1, 0])).toEqual(2);
        expect(a.locToIndex([1, 1])).toEqual(3);
        expect(a.locToIndex([2, 0])).toEqual(4);
        expect(a.locToIndex([2, 1])).toEqual(5);
        var b = dl.zeros([3, 2]);
        expect(b.locToIndex([0, 0])).toEqual(0);
        expect(b.locToIndex([0, 1])).toEqual(1);
        expect(b.locToIndex([1, 0])).toEqual(2);
        expect(b.locToIndex([1, 1])).toEqual(3);
        expect(b.locToIndex([2, 0])).toEqual(4);
        expect(b.locToIndex([2, 1])).toEqual(5);
    });
    it('locToIndex Array3D', function () {
        var a = dl.zeros([3, 2, 2]);
        expect(a.locToIndex([0, 0, 0])).toEqual(0);
        expect(a.locToIndex([0, 0, 1])).toEqual(1);
        expect(a.locToIndex([0, 1, 0])).toEqual(2);
        expect(a.locToIndex([0, 1, 1])).toEqual(3);
        expect(a.locToIndex([1, 0, 0])).toEqual(4);
        expect(a.locToIndex([1, 0, 1])).toEqual(5);
        expect(a.locToIndex([2, 1, 1])).toEqual(11);
        var b = dl.zeros([3, 2, 2]);
        expect(b.locToIndex([0, 0, 0])).toEqual(0);
        expect(b.locToIndex([0, 0, 1])).toEqual(1);
        expect(b.locToIndex([0, 1, 0])).toEqual(2);
        expect(b.locToIndex([0, 1, 1])).toEqual(3);
        expect(b.locToIndex([1, 0, 0])).toEqual(4);
        expect(b.locToIndex([1, 0, 1])).toEqual(5);
        expect(b.locToIndex([2, 1, 1])).toEqual(11);
    });
    it('NDArray<math>', function () {
        var a = null;
        var b = a;
        expect(b).toBeNull();
        var a1 = null;
        var b1 = a1;
        expect(b1).toBeNull();
        var a2 = null;
        var b2 = a2;
        expect(b2).toBeNull();
        var a3 = null;
        var b3 = a3;
        expect(b3).toBeNull();
        var a4 = null;
        var b4 = a4;
        expect(b4).toBeNull();
    });
};
var testsNew = function (it) {
    it('Array1D.new() from number[]', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        test_util.expectArraysClose(a, [1, 2, 3]);
    });
    it('Array1D.new() from number[][], shape mismatch', function () {
        expect(function () { return ndarray_1.Array1D.new([[1], [2], [3]]); }).toThrowError();
    });
    it('Array2D.new() from number[][]', function () {
        var a = ndarray_1.Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
        test_util.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
    });
    it('Array2D.new() from number[][], but shape does not match', function () {
        expect(function () { return ndarray_1.Array2D.new([3, 2], [[1, 2, 3], [4, 5, 6]]); }).toThrowError();
    });
    it('Array3D.new() from number[][][]', function () {
        var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[4], [5], [6]]]);
        test_util.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
    });
    it('Array3D.new() from number[][][], but shape does not match', function () {
        var values = [[[1], [2], [3]], [[4], [5], [6]]];
        expect(function () { return ndarray_1.Array3D.new([3, 2, 1], values); }).toThrowError();
    });
    it('Array4D.new() from number[][][][]', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
        test_util.expectArraysClose(a, [1, 2, 4, 5]);
    });
    it('Array4D.new() from number[][][][], but shape does not match', function () {
        var f = function () {
            ndarray_1.Array4D.new([2, 1, 2, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
        };
        expect(f).toThrowError();
    });
};
var testsFill = function (it) {
    it('1D fill', function () {
        var a = dl.zeros([3]);
        a.fill(2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [2, 2, 2]);
    });
    it('2D fill', function () {
        var a = dl.zeros([3, 2]);
        a.fill(2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
    });
    it('3D fill', function () {
        var a = dl.zeros([3, 2, 1]);
        a.fill(2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1]);
        test_util.expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
    });
    it('4D fill', function () {
        var a = dl.zeros([3, 2, 1, 2]);
        a.fill(2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 2]);
        test_util.expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    });
};
var testsScalarNew = function (it) {
    it('default dtype', function () {
        var a = ndarray_1.Scalar.new(3);
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [3]);
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Scalar.new(3, 'float32');
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [3]);
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Scalar.new(3, 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [3]);
    });
    it('int32 dtype, 3.9 => 3, like numpy', function () {
        var a = ndarray_1.Scalar.new(3.9, 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [3]);
    });
    it('int32 dtype, -3.9 => -3, like numpy', function () {
        var a = ndarray_1.Scalar.new(-3.9, 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [-3]);
    });
    it('bool dtype, 3 => true, like numpy', function () {
        var a = ndarray_1.Scalar.new(3, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('bool dtype, -2 => true, like numpy', function () {
        var a = ndarray_1.Scalar.new(-2, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('bool dtype, 0 => false, like numpy', function () {
        var a = ndarray_1.Scalar.new(0, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(0);
    });
    it('bool dtype from boolean', function () {
        var a = ndarray_1.Scalar.new(false, 'bool');
        expect(a.get()).toBe(0);
        expect(a.dtype).toBe('bool');
        var b = ndarray_1.Scalar.new(true, 'bool');
        expect(b.get()).toBe(1);
        expect(b.dtype).toBe('bool');
    });
    it('int32 dtype from boolean', function () {
        var a = ndarray_1.Scalar.new(true, 'int32');
        expect(a.get()).toBe(1);
        expect(a.dtype).toBe('int32');
    });
    it('default dtype from boolean', function () {
        var a = ndarray_1.Scalar.new(false);
        test_util.expectNumbersClose(a.get(), 0);
        expect(a.dtype).toBe('float32');
    });
};
var testsArray1DNew = function (it) {
    it('default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [1, 2, 3]);
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [1, 2, 3]);
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [1, 2, 3]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array1D.new([1.1, 2.5, 3.9], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [1, 2, 3]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array1D.new([-1.1, -2.5, -3.9], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [-1, -2, -3]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array1D.new([1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([4]);
        expect(a.get(0)).toBe(1);
        expect(a.get(1)).toBe(1);
        expect(a.get(2)).toBe(0);
        expect(a.get(3)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array1D.new([false, false, true]);
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [0, 0, 1]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array1D.new([false, false, true], 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [0, 0, 1]);
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array1D.new([false, false, true], 'bool');
        expect(a.dtype).toBe('bool');
        test_util.expectArraysEqual(a, [0, 0, 1]);
    });
};
var testsArray2DNew = function (it) {
    it('default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2]);
        test_util.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2]);
        test_util.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[1, 2], [3, 4]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1.1, 2.5, 3.9, 4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array2D.new([2, 2], [-1.1, -2.5, -3.9, -4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2]);
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(0);
        expect(a.get(1, 1)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[false, false], [true, false]]);
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[false, false], [true, false]], 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[false, false], [true, false]], 'bool');
        expect(a.dtype).toBe('bool');
        test_util.expectArraysEqual(a, [0, 0, 1, 0]);
    });
};
var testsArray3DNew = function (it) {
    it('default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[1], [2]], [[3], [4]]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.get(0, 0, 0)).toBe(1);
        expect(a.get(0, 1, 0)).toBe(1);
        expect(a.get(1, 0, 0)).toBe(0);
        expect(a.get(1, 1, 0)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]]);
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]], 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]], 'bool');
        expect(a.dtype).toBe('bool');
        test_util.expectArraysEqual(a, [0, 0, 1, 0]);
    });
};
var testsArray4DNew = function (it) {
    it('default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[3]], [[4]]]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(a, [-1, -2, -3, -4]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.get(0, 0, 0, 0)).toBe(1);
        expect(a.get(0, 1, 0, 0)).toBe(1);
        expect(a.get(1, 0, 0, 0)).toBe(0);
        expect(a.get(1, 1, 0, 0)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]]);
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'int32');
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'bool');
        expect(a.dtype).toBe('bool');
        test_util.expectArraysEqual(a, [0, 0, 1, 0]);
    });
};
var testsReshape = function (it) {
    it('Scalar default dtype', function () {
        var a = ndarray_1.Scalar.new(4);
        var b = a.reshape([1, 1]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('Scalar bool dtype', function () {
        var a = ndarray_1.Scalar.new(4, 'bool');
        var b = a.reshape([1, 1, 1]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([1, 1, 1]);
    });
    it('Array1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3, 4]);
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('Array1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3, 4], 'int32');
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('Array2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
    });
    it('Array2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6], 'bool');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([6]);
    });
    it('Array3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
    });
    it('Array3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6], 'bool');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([6]);
    });
    it('Array4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([2, 3]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 3]);
    });
    it('Array4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6], 'int32');
        var b = a.reshape([3, 2]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3, 2]);
    });
    it('reshape is functional', function (math) {
        var a = ndarray_1.Scalar.new(2.4);
        var b = a.reshape([]);
        expect(a.id).not.toBe(b.id);
        b.dispose();
        test_util.expectArraysClose(a, [2.4]);
    });
};
var testsAsType = function (it) {
    it('scalar bool -> int32', function () {
        var a = ndarray_1.Scalar.new(true, 'bool').toInt();
        expect(a.dtype).toBe('int32');
        expect(a.get()).toBe(1);
    });
    it('array1d float32 -> int32', function () {
        var a = ndarray_1.Array1D.new([1.1, 3.9, -2.9, 0]).toInt();
        expect(a.dtype).toBe('int32');
        test_util.expectArraysEqual(a, [1, 3, -2, 0]);
    });
    it('array2d float32 -> bool', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1.1, 3.9, -2.9, 0]).asType(types_1.DType.bool);
        expect(a.dtype).toBe('bool');
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(1);
        expect(a.get(1, 1)).toBe(0);
    });
    it('array2d int32 -> bool', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 3, 0, -1], 'int32').toBool();
        expect(a.dtype).toBe('bool');
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(0);
        expect(a.get(1, 1)).toBe(1);
    });
    it('array3d bool -> float32', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [true, false, false, true], 'bool').toFloat();
        expect(a.dtype).toBe('float32');
        test_util.expectArraysClose(a, [1, 0, 0, 1]);
    });
    it('bool CPU -> GPU -> CPU', function () {
        var a = ndarray_1.Array1D.new([1, 2, 0, 0, 5], 'bool');
        test_util.expectArraysEqual(a, [1, 1, 0, 0, 1]);
    });
    it('int32 CPU -> GPU -> CPU', function () {
        var a = ndarray_1.Array1D.new([1, 2, 0, 0, 5], 'int32');
        test_util.expectArraysEqual(a, [1, 2, 0, 0, 5]);
    });
    it('asType is functional', function (math) {
        var a = ndarray_1.Scalar.new(2.4, 'float32');
        var b = a.toFloat();
        expect(a.id).not.toBe(b.id);
        b.dispose();
        test_util.expectArraysClose(a, [2.4]);
    });
};
var testSqueeze = function (it) {
    it('squeeze no axis', function () {
        var a = ndarray_1.Array2D.new([3, 1], [4, 2, 1], 'bool');
        var b = a.squeeze();
        expect(b.shape).toEqual([3]);
    });
    it('squeeze with axis', function () {
        var a = ndarray_1.Array3D.new([3, 1, 1], [4, 2, 1], 'bool');
        var b = a.squeeze([1]);
        expect(b.shape).toEqual([3, 1]);
    });
    it('squeeze wrong axis', function () {
        var a = ndarray_1.Array3D.new([3, 1, 1], [4, 2, 1], 'bool');
        expect(function () { return a.squeeze([0, 1]); }).toThrowError('axis 0 is not 1');
    });
};
var testsAsXD = function (it) {
    it('scalar -> 2d', function () {
        var a = ndarray_1.Scalar.new(4, 'int32');
        var b = a.as2D(1, 1);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('1d -> 2d', function () {
        var a = ndarray_1.Array1D.new([4, 2, 1], 'bool');
        var b = a.as2D(3, 1);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3, 1]);
    });
    it('2d -> 4d', function () {
        var a = ndarray_1.Array2D.new([2, 2], [4, 2, 1, 3]);
        var b = a.as4D(1, 1, 2, 2);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1, 2, 2]);
    });
    it('3d -> 2d', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [4, 2, 1, 3], 'float32');
        var b = a.as2D(2, 2);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('4d -> 1d', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [4, 2, 1, 3], 'bool');
        var b = a.as1D();
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([4]);
    });
};
var allTests = [
    tests, testsNew, testsFill, testsScalarNew, testsArray1DNew, testsArray2DNew,
    testsArray3DNew, testsArray4DNew, testsReshape, testsAsType, testsAsXD,
    testSqueeze
];
test_util.describeMathCPU('NDArray', allTests);
test_util.describeMathGPU('NDArray', allTests, [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=ndarray_test.js.map