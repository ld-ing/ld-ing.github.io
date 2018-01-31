"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var dl = require("../index");
var ndarray_1 = require("../math/ndarray");
var graph_1 = require("./graph");
var tensor_array_map_1 = require("./tensor_array_map");
describe('TensorArrayMap.size', function () {
    it('is 0 at construction', function () {
        expect((new tensor_array_map_1.TensorArrayMap()).size()).toEqual(0);
    });
    it('is 1 after add', function () {
        var map = new tensor_array_map_1.TensorArrayMap();
        map.set(new graph_1.Tensor([]), dl.zeros([1]));
        expect(map.size()).toEqual(1);
    });
    it('increments for every add', function () {
        var map = new tensor_array_map_1.TensorArrayMap();
        for (var i = 0; i < 9; ++i) {
            map.set(new graph_1.Tensor([]), dl.zeros([1]));
        }
        expect(map.size()).toEqual(9);
    });
});
describe('TensorArrayMap.hasNullArray', function () {
    var map;
    var t;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
        t = new graph_1.Tensor([]);
    });
    it('returns true for null NDArray entries', function () {
        map.set(t, null);
        expect(map.hasNullArray(t)).toBe(true);
    });
    it('returns false for non-null NDArray entries', function () {
        map.set(t, dl.zeros([1]));
        expect(map.hasNullArray(t)).toBe(false);
    });
    it('throws for missing keys', function () {
        expect(function () { return map.hasNullArray(t); }).toThrowError(/not in array map/);
    });
});
describe('TensorArrayMap.get', function () {
    var map;
    var t;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
        t = new graph_1.Tensor([]);
    });
    it('returns the associated NDArray', function () {
        var nda = dl.zeros([1]);
        map.set(t, nda);
        expect(map.get(t)).toBe(nda);
    });
    it('throws if associated NDArray is null', function () {
        map.set(t, null);
        expect(function () { return map.get(t); }).toThrowError(/has null array/);
    });
    it('throws for missing key', function () {
        expect(function () { return map.get(t); }).toThrowError(/not in array map/);
    });
});
describe('TensorArrayMap.delete', function () {
    var map;
    var t;
    beforeEach(function () {
        map = new tensor_array_map_1.TensorArrayMap();
        t = new graph_1.Tensor([]);
    });
    it('deletes the key from the map', function () {
        map.set(t, null);
        map.delete(t);
        expect(function () { return map.get(t); }).toThrow();
    });
    it('is tolerant of deleting nonexistent keys', function () {
        map.set(t, null);
        map.delete(t);
        map.delete(t);
        map.delete(t);
        map.delete(t);
    });
});
describe('SummedTensorArrayMap.add', function () {
    var map;
    var t;
    var math = environment_1.ENV.math;
    beforeEach(function () {
        map = new tensor_array_map_1.SummedTensorArrayMap(math);
        t = new graph_1.Tensor([]);
    });
    it('add sums gradients', function () {
        map.add(t, ndarray_1.Array1D.new([1, 2, 3]));
        expect(map.get(t).dataSync()).toEqual(new Float32Array([1, 2, 3]));
        map.add(t, ndarray_1.Array1D.new([30, 20, 10]));
        expect(map.get(t).dataSync()).toEqual(new Float32Array([31, 22, 13]));
    });
});
//# sourceMappingURL=tensor_array_map_test.js.map