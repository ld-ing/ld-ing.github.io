"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
require("../test_env");
var checkpoint_loader_1 = require("./checkpoint_loader");
describe('Checkpoint var loader', function () {
    var xhrObj;
    beforeEach(function () {
        xhrObj = jasmine.createSpyObj('xhrObj', ['addEventListener', 'open', 'send', 'onload', 'onerror']);
        spyOn(window, 'XMLHttpRequest').and.returnValue(xhrObj);
    });
    it('Load manifest and a variable', function (doneFn) {
        var fakeCheckpointManifest = {
            'fakeVar1': { filename: 'fakeFile1', shape: [10] },
            'fakeVar2': { filename: 'fakeFile2', shape: [5, 5] }
        };
        var varLoader = new checkpoint_loader_1.CheckpointLoader('fakeModel');
        varLoader.getCheckpointManifest().then(function (checkpoint) {
            expect(checkpoint).toEqual(fakeCheckpointManifest);
            var buffer = new ArrayBuffer(4 * fakeCheckpointManifest['fakeVar1'].shape[0]);
            var view = new Float32Array(buffer);
            for (var i = 0; i < 10; i++) {
                view[i] = i;
            }
            varLoader.getVariable('fakeVar1').then(function (ndarray) {
                expect(ndarray.shape).toEqual(fakeCheckpointManifest['fakeVar1'].shape);
                expect(ndarray.dataSync()).toEqual(view);
                doneFn();
            });
            xhrObj.response = buffer;
            xhrObj.onload();
        });
        xhrObj.responseText = JSON.stringify(fakeCheckpointManifest);
        xhrObj.onload();
    });
    it('Load manifest error', function () {
        var varLoader = new checkpoint_loader_1.CheckpointLoader('fakeModel');
        varLoader.getCheckpointManifest();
        expect(function () { return xhrObj.onerror(); }).toThrowError();
    });
    it('Load non-existent variable throws error', function (doneFn) {
        var fakeCheckpointManifest = { 'fakeVar1': { filename: 'fakeFile1', shape: [10] } };
        var varLoader = new checkpoint_loader_1.CheckpointLoader('fakeModel');
        varLoader.getCheckpointManifest().then(function (checkpoint) {
            expect(function () { return varLoader.getVariable('varDoesntExist'); }).toThrowError();
            doneFn();
        });
        xhrObj.responseText = JSON.stringify(fakeCheckpointManifest);
        xhrObj.onload();
    });
    it('Load variable throws error', function (doneFn) {
        var fakeCheckpointManifest = { 'fakeVar1': { filename: 'fakeFile1', shape: [10] } };
        var varLoader = new checkpoint_loader_1.CheckpointLoader('fakeModel');
        varLoader.getCheckpointManifest().then(function (checkpoint) {
            varLoader.getVariable('fakeVar1');
            expect(function () { return xhrObj.onerror(); }).toThrowError();
            doneFn();
        });
        xhrObj.responseText = JSON.stringify(fakeCheckpointManifest);
        xhrObj.onload();
    });
    it('Load variable but 404 not found', function (doneFn) {
        var fakeCheckpointManifest = { 'fakeVar1': { filename: 'fakeFile1', shape: [10] } };
        var varLoader = new checkpoint_loader_1.CheckpointLoader('fakeModel');
        varLoader.getCheckpointManifest().then(function (checkpoint) {
            varLoader.getVariable('fakeVar1');
            expect(function () { return xhrObj.onload(); }).toThrowError();
            doneFn();
        });
        xhrObj.responseText = JSON.stringify(fakeCheckpointManifest);
        xhrObj.status = 404;
        xhrObj.onload();
    });
});
//# sourceMappingURL=checkpoint_loader_test.js.map