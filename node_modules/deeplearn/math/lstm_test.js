"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dl = require("../index");
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('MultiRNNCell with 2 BasicLSTMCells', function (math) {
            var lstmKernel1 = ndarray_1.Array2D.new([3, 4], [
                0.26242125034332275, -0.8787832260131836, 0.781475305557251,
                1.337337851524353, 0.6180247068405151, -0.2760246992111206,
                -0.11299663782119751, -0.46332040429115295, -0.1765323281288147,
                0.6807947158813477, -0.8326982855796814, 0.6732975244522095
            ]);
            var lstmBias1 = ndarray_1.Array1D.new([1.090713620185852, -0.8282332420349121, 0, 1.0889357328414917]);
            var lstmKernel2 = ndarray_1.Array2D.new([2, 4], [
                -1.893059492111206, -1.0185645818710327, -0.6270437240600586,
                -2.1829540729522705, -0.4583775997161865, -0.5454602241516113,
                -0.3114445209503174, 0.8450229167938232
            ]);
            var lstmBias2 = ndarray_1.Array1D.new([0.9906240105628967, 0.6248329877853394, 0, 1.0224634408950806]);
            var forgetBias = ndarray_1.Scalar.new(1.0);
            var lstm1 = math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
            var lstm2 = math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);
            var c = [
                dl.zeros([1, lstmBias1.shape[0] / 4]),
                dl.zeros([1, lstmBias2.shape[0] / 4])
            ];
            var h = [
                dl.zeros([1, lstmBias1.shape[0] / 4]),
                dl.zeros([1, lstmBias2.shape[0] / 4])
            ];
            var onehot = dl.zeros([1, 2]);
            onehot.set(1.0, 0, 0);
            var output = math.multiRNNCell([lstm1, lstm2], onehot, c, h);
            test_util.expectArraysClose(output[0][0], [-0.7440074682235718]);
            test_util.expectArraysClose(output[0][1], [0.7460772395133972]);
            test_util.expectArraysClose(output[1][0], [-0.5802832245826721]);
            test_util.expectArraysClose(output[1][1], [0.5745711922645569]);
        });
        it('basicLSTMCell with batch=2', function (math) {
            var lstmKernel = dl.randNormal([3, 4]);
            var lstmBias = dl.randNormal([4]);
            var forgetBias = ndarray_1.Scalar.new(1.0);
            var data = dl.randNormal([1, 2]);
            var batchedData = math.concat2D(data, data, 0);
            var c = dl.randNormal([1, 1]);
            var batchedC = math.concat2D(c, c, 0);
            var h = dl.randNormal([1, 1]);
            var batchedH = math.concat2D(h, h, 0);
            var _a = math.basicLSTMCell(forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH), newC = _a[0], newH = _a[1];
            expect(newC.get(0, 0)).toEqual(newC.get(1, 0));
            expect(newH.get(0, 0)).toEqual(newH.get(1, 0));
        });
    };
    test_util.describeMathCPU('basicLSTMCell', [tests]);
    test_util.describeMathGPU('basicLSTMCell', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=lstm_test.js.map