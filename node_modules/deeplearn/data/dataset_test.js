"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var ndarray_1 = require("../math/ndarray");
var test_util = require("../test_util");
var dataset_1 = require("./dataset");
var StubDataset = (function (_super) {
    __extends(StubDataset, _super);
    function StubDataset(data) {
        var _this = _super.call(this, data.map(function (value) { return value[0].shape; })) || this;
        _this.dataset = data;
        return _this;
    }
    StubDataset.prototype.fetchData = function () {
        return new Promise(function (resolve, reject) { });
    };
    return StubDataset;
}(dataset_1.InMemoryDataset));
describe('Dataset', function () {
    it('normalize', function () {
        var data = [
            [
                ndarray_1.Array2D.new([2, 3], [1, 2, 10, -1, -2, .75]),
                ndarray_1.Array2D.new([2, 3], [2, 3, 20, -2, 2, .5]),
                ndarray_1.Array2D.new([2, 3], [3, 4, 30, -3, -4, 0]),
                ndarray_1.Array2D.new([2, 3], [4, 5, 40, -4, 4, 1])
            ],
            [
                dl.randNormal([1]), dl.randNormal([1]), dl.randNormal([1]),
                dl.randNormal([1])
            ]
        ];
        var dataset = new StubDataset(data);
        var dataIndex = 0;
        dataset.normalizeWithinBounds(dataIndex, 0, 1);
        var normalizedInputs = dataset.getData()[0];
        test_util.expectArraysClose(normalizedInputs[0], [0, 0, 0, 1, .25, .75]);
        test_util.expectArraysClose(normalizedInputs[1], [1 / 3, 1 / 3, 1 / 3, 2 / 3, .75, .5]);
        test_util.expectArraysClose(normalizedInputs[2], [2 / 3, 2 / 3, 2 / 3, 1 / 3, 0, 0]);
        test_util.expectArraysClose(normalizedInputs[3], [1, 1, 1, 0, 1, 1]);
        dataset.normalizeWithinBounds(dataIndex, -1, 1);
        normalizedInputs = dataset.getData()[0];
        test_util.expectArraysClose(normalizedInputs[0], [-1, -1, -1, 1, -.5, .5]);
        test_util.expectArraysClose(normalizedInputs[1], [-1 / 3, -1 / 3, -1 / 3, 1 / 3, .5, .0]);
        test_util.expectArraysClose(normalizedInputs[2], [1 / 3, 1 / 3, 1 / 3, -1 / 3, -1, -1]);
        test_util.expectArraysClose(normalizedInputs[3], [1, 1, 1, -1, 1, 1]);
        dataset.removeNormalization(dataIndex);
        normalizedInputs = dataset.getData()[0];
        test_util.expectArraysClose(normalizedInputs[0], [1, 2, 10, -1, -2, .75]);
        test_util.expectArraysClose(normalizedInputs[1], [2, 3, 20, -2, 2, .5]);
        test_util.expectArraysClose(normalizedInputs[2], [3, 4, 30, -3, -4, 0]);
        test_util.expectArraysClose(normalizedInputs[3], [4, 5, 40, -4, 4, 1]);
    });
});
//# sourceMappingURL=dataset_test.js.map