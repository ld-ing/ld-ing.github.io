"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
var testsZeros = function (it) {
    it('1D default dtype', function () {
        var a = dl.zeros([3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [0, 0, 0]);
    });
    it('1D float32 dtype', function () {
        var a = dl.zeros([3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [0, 0, 0]);
    });
    it('1D int32 dtype', function () {
        var a = dl.zeros([3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [0, 0, 0]);
    });
    it('1D bool dtype', function () {
        var a = dl.zeros([3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [0, 0, 0]);
    });
    it('2D default dtype', function () {
        var a = dl.zeros([3, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('2D float32 dtype', function () {
        var a = dl.zeros([3, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('2D int32 dtype', function () {
        var a = dl.zeros([3, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
    it('2D bool dtype', function () {
        var a = dl.zeros([3, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
    it('3D default dtype', function () {
        var a = dl.zeros([2, 2, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('3D float32 dtype', function () {
        var a = dl.zeros([2, 2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('3D int32 dtype', function () {
        var a = dl.zeros([2, 2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('3D bool dtype', function () {
        var a = dl.zeros([2, 2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('4D default dtype', function () {
        var a = dl.zeros([3, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('4D float32 dtype', function () {
        var a = dl.zeros([3, 2, 1, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('4D int32 dtype', function () {
        var a = dl.zeros([3, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
    it('4D bool dtype', function () {
        var a = dl.zeros([3, 2, 1, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
};
var testsOnes = function (it) {
    it('1D default dtype', function () {
        var a = dl.ones([3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [1, 1, 1]);
    });
    it('1D float32 dtype', function () {
        var a = dl.ones([3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysClose(a, [1, 1, 1]);
    });
    it('1D int32 dtype', function () {
        var a = dl.ones([3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [1, 1, 1]);
    });
    it('1D bool dtype', function () {
        var a = dl.ones([3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3]);
        test_util.expectArraysEqual(a, [1, 1, 1]);
    });
    it('2D default dtype', function () {
        var a = dl.ones([3, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('2D float32 dtype', function () {
        var a = dl.ones([3, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('2D int32 dtype', function () {
        var a = dl.ones([3, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
    it('2D bool dtype', function () {
        var a = dl.ones([3, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2]);
        test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
    it('3D default dtype', function () {
        var a = dl.ones([2, 2, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('3D float32 dtype', function () {
        var a = dl.ones([2, 2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('3D int32 dtype', function () {
        var a = dl.ones([2, 2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('3D bool dtype', function () {
        var a = dl.ones([2, 2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('4D default dtype', function () {
        var a = dl.ones([3, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('4D float32 dtype', function () {
        var a = dl.ones([3, 2, 1, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('4D int32 dtype', function () {
        var a = dl.ones([3, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
    it('4D bool dtype', function () {
        var a = dl.ones([3, 2, 1, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
};
var testsZerosLike = function (it) {
    it('1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysClose(b, [0, 0, 0]);
    });
    it('1D float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysClose(b, [0, 0, 0]);
    });
    it('1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysEqual(b, [0, 0, 0]);
    });
    it('1D bool dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'bool');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysEqual(b, [0, 0, 0]);
    });
    it('2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('2D float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('2D int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('3D float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('3D int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('4D float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('4D bool dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
        var b = dl.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(b, [0, 0, 0, 0]);
    });
};
var testsOnesLike = function (it) {
    it('1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysClose(b, [1, 1, 1]);
    });
    it('1D float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysClose(b, [1, 1, 1]);
    });
    it('1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysEqual(b, [1, 1, 1]);
    });
    it('1D bool dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'bool');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysEqual(b, [1, 1, 1]);
    });
    it('2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('2D float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('2D int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('3D float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('3D int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('4D float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('4D bool dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
        var b = dl.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
};
var testsRand = function (it) {
    it('should return a random 1D float32 array', function () {
        var shape = [10];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2); });
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2);
        result = dl.rand(shape, function () { return util.randUniform(0, 1.5); });
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 1D int32 array', function () {
        var shape = [10];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 1D bool array', function () {
        var shape = [10];
        var result = dl.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 2D float32 array', function () {
        var shape = [3, 4];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2.5); });
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.rand(shape, function () { return util.randUniform(0, 1.5); }, 'float32');
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 2D int32 array', function () {
        var shape = [3, 4];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 2D bool array', function () {
        var shape = [3, 4];
        var result = dl.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 3D float32 array', function () {
        var shape = [3, 4, 5];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2.5); });
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.rand(shape, function () { return util.randUniform(0, 1.5); }, 'float32');
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 3D int32 array', function () {
        var shape = [3, 4, 5];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 3D bool array', function () {
        var shape = [3, 4, 5];
        var result = dl.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 4D float32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2.5); });
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.rand(shape, function () { return util.randUniform(0, 1.5); });
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 4D int32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = dl.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 4D bool array', function () {
        var shape = [3, 4, 5, 6];
        var result = dl.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
};
var testsRandNormal = function (it) {
    var SEED = 2002;
    var EPSILON = 0.05;
    it('should return a float32 1D of random normal values', function () {
        var SAMPLES = 10000;
        var result = dl.randNormal([SAMPLES], 0, 0.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);
        result = dl.randNormal([SAMPLES], 0, 1.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
    });
    it('should return a int32 1D of random normal values', function () {
        var SAMPLES = 10000;
        var result = dl.randNormal([SAMPLES], 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a float32 2D of random normal values', function () {
        var SAMPLES = 250;
        var result = dl.randNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 2.5, EPSILON);
        result = dl.randNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
    });
    it('should return a int32 2D of random normal values', function () {
        var SAMPLES = 100;
        var result = dl.randNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a float32 3D of random normal values', function () {
        var SAMPLES = 50;
        var result = dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);
        result =
            dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
    });
    it('should return a int32 3D of random normal values', function () {
        var SAMPLES = 50;
        var result = dl.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a float32 4D of random normal values', function () {
        var SAMPLES = 25;
        var result = dl.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 0.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);
        result = dl.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
    });
    it('should return a int32 4D of random normal values', function () {
        var SAMPLES = 25;
        var result = dl.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
        test_util.jarqueBeraNormalityTest(result);
        test_util.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
};
var testsRandTruncNormal = function (it) {
    var EPSILON = 0.60;
    var SEED = 2002;
    function assertTruncatedValues(array, mean, stdv) {
        var bounds = mean + stdv * 2;
        var values = array.dataSync();
        for (var i = 0; i < values.length; i++) {
            expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
        }
    }
    it('should return a random 1D float32 array', function () {
        var shape = [1000];
        var result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a randon 1D int32 array', function () {
        var shape = [1000];
        var result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
    it('should return a 2D float32 array', function () {
        var shape = [50, 50];
        var result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a 2D int32 array', function () {
        var shape = [50, 50];
        var result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
    it('should return a 3D float32 array', function () {
        var shape = [10, 10, 10];
        var result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a 3D int32 array', function () {
        var shape = [10, 10, 10];
        var result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
    it('should return a 4D float32 array', function () {
        var shape = [5, 5, 5, 5];
        var result = dl.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        test_util.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = dl.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        test_util.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a 4D int32 array', function () {
        var shape = [5, 5, 5, 5];
        var result = dl.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        test_util.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
};
var testsRandUniform = function (it) {
    it('should return a random 1D float32 array', function () {
        var shape = [10];
        var result = dl.randUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.randUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 1D int32 array', function () {
        var shape = [10];
        var result = dl.randUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 1D bool array', function () {
        var shape = [10];
        var result = dl.randUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 2D float32 array', function () {
        var shape = [3, 4];
        var result = dl.randUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.randUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 2D int32 array', function () {
        var shape = [3, 4];
        var result = dl.randUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 2D bool array', function () {
        var shape = [3, 4];
        var result = dl.randUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 3D float32 array', function () {
        var shape = [3, 4, 5];
        var result = dl.randUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.randUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 3D int32 array', function () {
        var shape = [3, 4, 5];
        var result = dl.randUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 3D bool array', function () {
        var shape = [3, 4, 5];
        var result = dl.randUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 4D float32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = dl.randUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 2.5);
        result = dl.randUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 4D int32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = dl.randUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 4D bool array', function () {
        var shape = [3, 4, 5, 6];
        var result = dl.randUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util.expectValuesInRange(result, 0, 1);
    });
};
var testsFromPixels = function (it) {
    beforeEach(function () { });
    afterEach(function () { });
    it('ImageData 1x1x3', function () {
        var pixels = new ImageData(1, 1);
        pixels.data[0] = 0;
        pixels.data[1] = 80;
        pixels.data[2] = 160;
        pixels.data[3] = 240;
        var array = dl.fromPixels(pixels, 3);
        test_util.expectArraysEqual(array, [0, 80, 160]);
    });
    it('ImageData 1x1x4', function () {
        var pixels = new ImageData(1, 1);
        pixels.data[0] = 0;
        pixels.data[1] = 80;
        pixels.data[2] = 160;
        pixels.data[3] = 240;
        var array = dl.fromPixels(pixels, 4);
        test_util.expectArraysEqual(array, [0, 80, 160, 240]);
    });
    it('ImageData 2x2x3', function () {
        var pixels = new ImageData(2, 2);
        for (var i = 0; i < 8; i++) {
            pixels.data[i] = i * 2;
        }
        for (var i = 8; i < 16; i++) {
            pixels.data[i] = i * 2;
        }
        var array = dl.fromPixels(pixels, 3);
        test_util.expectArraysEqual(array, [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]);
    });
    it('ImageData 2x2x4', function () {
        var pixels = new ImageData(2, 2);
        for (var i = 0; i < 8; i++) {
            pixels.data[i] = i * 2;
        }
        for (var i = 8; i < 16; i++) {
            pixels.data[i] = i * 2;
        }
        var array = dl.fromPixels(pixels, 4);
        test_util.expectArraysClose(array, new Int32Array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
    });
    it('fromPixels, 3 channels', function () {
        var pixels = new ImageData(1, 2);
        pixels.data[0] = 2;
        pixels.data[1] = 3;
        pixels.data[2] = 4;
        pixels.data[3] = 255;
        pixels.data[4] = 5;
        pixels.data[5] = 6;
        pixels.data[6] = 7;
        pixels.data[7] = 255;
        var res = dl.fromPixels(pixels, 3);
        expect(res.shape).toEqual([2, 1, 3]);
        expect(res.dtype).toBe('int32');
        test_util.expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
    });
    it('fromPixels, reshape, then do dl.add()', function () {
        var pixels = new ImageData(1, 1);
        pixels.data[0] = 2;
        pixels.data[1] = 3;
        pixels.data[2] = 4;
        pixels.data[3] = 255;
        var a = dl.fromPixels(pixels, 3).reshape([1, 1, 1, 3]);
        var res = a.add(ndarray_1.Scalar.new(2, 'int32'));
        expect(res.shape).toEqual([1, 1, 1, 3]);
        expect(res.dtype).toBe('int32');
        test_util.expectArraysClose(res, [4, 5, 6]);
    });
    it('fromPixels + fromPixels', function () {
        var pixelsA = new ImageData(1, 1);
        pixelsA.data[0] = 255;
        pixelsA.data[1] = 3;
        pixelsA.data[2] = 4;
        pixelsA.data[3] = 255;
        var pixelsB = new ImageData(1, 1);
        pixelsB.data[0] = 5;
        pixelsB.data[1] = 6;
        pixelsB.data[2] = 7;
        pixelsB.data[3] = 255;
        var a = dl.fromPixels(pixelsA, 3).toFloat();
        var b = dl.fromPixels(pixelsB, 3).toFloat();
        var res = a.add(b);
        expect(res.shape).toEqual([1, 1, 3]);
        expect(res.dtype).toBe('float32');
        test_util.expectArraysClose(res, [260, 9, 11]);
    });
};
var testsClone = function (it) {
    it('1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysClose(b, [1, 2, 3]);
    });
    it('1D float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysClose(b, [1, 2, 3]);
    });
    it('1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysEqual(b, [1, 2, 3]);
    });
    it('1D bool dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'bool');
        var b = dl.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        test_util.expectArraysEqual(b, [1, 1, 1]);
    });
    it('2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('2D float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('2D int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(b, [1, 2, 3, 4]);
    });
    it('2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
        var b = dl.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('3D float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('3D int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(b, [1, 2, 3, 4]);
    });
    it('3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
        var b = dl.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('4D float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
        var b = dl.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(b, [1, 2, 3, 4]);
    });
    it('4D bool dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
        var b = dl.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util.expectArraysEqual(b, [1, 1, 1, 1]);
    });
};
var allTests = [
    testsZeros,
    testsOnes,
    testsZerosLike,
    testsOnesLike,
    testsClone,
    testsRand,
    testsRandNormal,
    testsRandTruncNormal,
    testsRandUniform,
    testsFromPixels,
];
test_util.describeMathCPU('array_ops', allTests);
test_util.describeMathGPU('array_ops', allTests, [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=array_ops_test.js.map