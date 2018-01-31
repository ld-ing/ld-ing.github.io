"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var ndarray_1 = require("../math/ndarray");
var sgd_optimizer_1 = require("../math/optimizers/sgd_optimizer");
var graph_1 = require("./graph");
var graph_runner_1 = require("./graph_runner");
var session_1 = require("./session");
var FAKE_LEARNING_RATE = 1.0;
var FAKE_BATCH_SIZE = 10;
function fakeTrainBatch(math, feedEntries, batchSize, optimizer, costReduction) {
    return ndarray_1.Scalar.new(.5);
}
describe('Model runner', function () {
    var math = environment_1.ENV.math;
    var g;
    var session;
    var optimizer;
    var inputTensor;
    var labelTensor;
    var costTensor;
    var predictionTensor;
    var metricTensor;
    var graphRunner;
    var avgCostCallback;
    var metricCallback;
    var originalTimeout;
    var fakeUserEvents = {
        batchesTrainedCallback: function (totalBatchesTrained) { return null; },
        avgCostCallback: function (avgCost) { return avgCostCallback(avgCost); },
        metricCallback: function (metric) { return metricCallback(metric); },
        inferenceExamplesCallback: function (feeds, inferenceValues) { return null; },
        trainExamplesPerSecCallback: function (examplesPerSec) { return null; },
        totalTimeCallback: function (totalTime) { return null; }
    };
    beforeEach(function () {
        originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
        jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;
        g = new graph_1.Graph();
        optimizer = new sgd_optimizer_1.SGDOptimizer(FAKE_LEARNING_RATE);
        inputTensor = g.placeholder('input', [2]);
        predictionTensor = g.add(inputTensor, g.constant(ndarray_1.Array1D.new([1, 1])));
        labelTensor = g.placeholder('label', [2]);
        costTensor = g.softmaxCrossEntropyCost(predictionTensor, labelTensor);
        metricTensor = g.argmaxEquals(predictionTensor, labelTensor);
        session = new session_1.Session(g, math);
        spyOn(session, 'train').and.callFake(fakeTrainBatch);
        var counter = 0;
        spyOn(session, 'eval').and.callFake(function (evalTensor) {
            if (evalTensor === predictionTensor) {
                return ndarray_1.Array1D.new([1, 0]);
            }
            else if (evalTensor === metricTensor) {
                return ndarray_1.Scalar.new(counter++ % 2);
            }
            else {
                throw new Error('Eval tensor not recognized');
            }
        });
        spyOn(fakeUserEvents, 'batchesTrainedCallback').and.callThrough();
        spyOn(fakeUserEvents, 'avgCostCallback').and.callThrough();
        spyOn(fakeUserEvents, 'metricCallback').and.callThrough();
        spyOn(fakeUserEvents, 'inferenceExamplesCallback').and.callThrough();
        spyOn(fakeUserEvents, 'trainExamplesPerSecCallback').and.callThrough();
        spyOn(fakeUserEvents, 'totalTimeCallback').and.callThrough();
    });
    afterEach(function () {
        jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
    });
    it('basic train usage, train 3 batches', function (doneFn) {
        var numBatches = 3;
        var trainFeedEntries = [];
        var testExampleProvider = [];
        fakeUserEvents.doneTrainingCallback = function () {
            for (var i = 0; i < numBatches; i++) {
                var args = session.train.calls.argsFor(i);
                expect(args).toEqual([
                    costTensor, trainFeedEntries, FAKE_BATCH_SIZE, optimizer,
                    session_1.CostReduction.MEAN
                ]);
                fakeUserEvents.avgCostCallback.calls.argsFor(i);
                fakeUserEvents.metricCallback.calls.argsFor(i);
            }
            expect(fakeUserEvents.avgCostCallback.calls.count())
                .toEqual(numBatches);
            expect(fakeUserEvents.metricCallback.calls.count())
                .toEqual(numBatches);
            expect(session.train.calls.count()).toEqual(numBatches);
            expect(session.eval.calls.count())
                .toEqual(FAKE_BATCH_SIZE * numBatches);
            expect(fakeUserEvents.avgCostCallback.calls.count())
                .toEqual(numBatches);
            expect(fakeUserEvents.metricCallback.calls.count())
                .toEqual(numBatches);
            expect(fakeUserEvents.trainExamplesPerSecCallback
                .calls.count())
                .toEqual(numBatches);
            expect(fakeUserEvents.totalTimeCallback.calls.count())
                .toEqual(numBatches);
            expect(fakeUserEvents.batchesTrainedCallback.calls.count())
                .toEqual(numBatches);
            expect(graphRunner.getTotalBatchesTrained()).toEqual(numBatches);
            expect(fakeUserEvents.inferenceExamplesCallback
                .calls.count())
                .toEqual(0);
            doneFn();
        };
        avgCostCallback = function (avgCost) {
            expect(avgCost.get()).toEqual(.5);
        };
        metricCallback = function (metric) {
            expect(metric.get()).toEqual(.5);
        };
        graphRunner = new graph_runner_1.GraphRunner(math, session, fakeUserEvents);
        expect(graphRunner.getTotalBatchesTrained()).toEqual(0);
        graphRunner.train(costTensor, trainFeedEntries, FAKE_BATCH_SIZE, optimizer, numBatches, metricTensor, testExampleProvider, FAKE_BATCH_SIZE, graph_runner_1.MetricReduction.MEAN, 0, 0);
    });
    it('basic inference usage', function (doneFn) {
        var intervalMs = 0;
        var exampleCount = 2;
        var numPasses = 1;
        fakeUserEvents.inferenceExamplesCallback =
            function (inputInferenceExamples, inferenceOutputs) {
                expect(inputInferenceExamples.length).toEqual(exampleCount);
                expect(inferenceOutputs.length).toEqual(exampleCount);
                expect(session.eval.calls.count())
                    .toEqual(exampleCount * numPasses);
                expect(graphRunner.getTotalBatchesTrained()).toEqual(0);
                expect(fakeUserEvents.avgCostCallback.calls.count())
                    .toEqual(0);
                expect(fakeUserEvents.metricCallback.calls.count())
                    .toEqual(0);
                expect(fakeUserEvents.trainExamplesPerSecCallback
                    .calls.count())
                    .toEqual(0);
                expect(fakeUserEvents.totalTimeCallback.calls.count())
                    .toEqual(0);
                expect(fakeUserEvents.batchesTrainedCallback
                    .calls.count())
                    .toEqual(0);
                expect(graphRunner.getTotalBatchesTrained()).toEqual(0);
                doneFn();
            };
        graphRunner = new graph_runner_1.GraphRunner(math, session, fakeUserEvents);
        var inferenceFeedEntries = [];
        graphRunner.infer(predictionTensor, inferenceFeedEntries, intervalMs, exampleCount, numPasses);
    });
});
//# sourceMappingURL=graph_runner_test.js.map