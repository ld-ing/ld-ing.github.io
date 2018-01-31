"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var priority_queue = require("./priority_queue");
var priority_queue_1 = require("./priority_queue");
describe('defaultCompare', function () {
    it('returns 0 if a === b', function () {
        expect(priority_queue.defaultCompare(123, 123)).toEqual(0);
    });
    it('returns 1 if a > b', function () {
        expect(priority_queue.defaultCompare(1000, 999)).toEqual(1);
    });
    it('returns -1 if a < b', function () {
        expect(priority_queue.defaultCompare(999, 1000)).toEqual(-1);
    });
});
describe('PriorityQueue', function () {
    var pq;
    beforeEach(function () {
        pq = new priority_queue_1.PriorityQueue(priority_queue.defaultCompare);
    });
    it('is empty by default', function () {
        expect(pq.empty()).toEqual(true);
    });
    it('isn\'t empty after enqueue call', function () {
        pq.enqueue(0);
        expect(pq.empty()).toEqual(false);
    });
    it('returns to empty after dequeueing only element', function () {
        pq.enqueue(0);
        pq.dequeue();
        expect(pq.empty()).toEqual(true);
    });
    it('returns to empty after dequeueing last element', function () {
        for (var i = 0; i < 10; ++i) {
            pq.enqueue(i);
        }
        for (var i = 0; i < 9; ++i) {
            pq.dequeue();
            expect(pq.empty()).toEqual(false);
        }
        pq.dequeue();
        expect(pq.empty()).toEqual(true);
    });
    it('dequeue throws when queue is empty', function () {
        expect(function () { return pq.dequeue(); })
            .toThrow(new Error('dequeue called on empty priority queue.'));
    });
    it('dequeues the only enqueued item', function () {
        pq.enqueue(1);
        expect(pq.dequeue()).toEqual(1);
    });
    it('dequeues the lowest-priority of 2 items', function () {
        pq.enqueue(1000);
        pq.enqueue(0);
        expect(pq.dequeue()).toEqual(0);
    });
    it('dequeues items in min-priority order', function () {
        pq.enqueue(5);
        pq.enqueue(8);
        pq.enqueue(2);
        pq.enqueue(9);
        pq.enqueue(3);
        pq.enqueue(7);
        pq.enqueue(4);
        pq.enqueue(0);
        pq.enqueue(6);
        pq.enqueue(1);
        var dequeued = [];
        while (!pq.empty()) {
            dequeued.push(pq.dequeue());
        }
        expect(dequeued).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });
});
describe('PriorityQueue index observer', function () {
    var pq;
    var indices;
    beforeEach(function () {
        pq = new priority_queue_1.PriorityQueue(priority_queue.defaultCompare, function (value, newIndex) { return indices[value] = newIndex; });
        indices = {};
    });
    it('notifies of new index when enqueuing', function () {
        pq.enqueue(0);
        expect(indices[0]).not.toBe(null);
    });
    it('puts first enqueued element at root of heap (index 0)', function () {
        pq.enqueue(0);
        expect(indices[0]).toEqual(0);
    });
    it('puts second greater element at left child of root (index 1)', function () {
        pq.enqueue(0);
        pq.enqueue(1);
        expect(indices[0]).toEqual(0);
        expect(indices[1]).toEqual(1);
    });
    it('puts third greater element at right child of root (index 2)', function () {
        pq.enqueue(0);
        pq.enqueue(1);
        pq.enqueue(2);
        expect(indices[0]).toEqual(0);
        expect(indices[1]).toEqual(1);
        expect(indices[2]).toEqual(2);
    });
    it('swaps root with new min enqueued element', function () {
        pq.enqueue(1000);
        pq.enqueue(0);
        expect(indices[1000]).toEqual(1);
        expect(indices[0]).toEqual(0);
    });
});
var TestEntry = (function () {
    function TestEntry(id, priority) {
        this.id = id;
        this.priority = priority;
    }
    return TestEntry;
}());
describe('PriorityQueue.update', function () {
    var pq;
    var indices;
    beforeEach(function () {
        pq = new priority_queue_1.PriorityQueue(function (a, b) {
            return priority_queue.defaultCompare(a.priority, b.priority);
        }, function (entry, newIndex) { return indices[entry.id] = newIndex; });
        indices = {};
    });
    it('no longer dequeues original min element after priority change', function () {
        var e0 = new TestEntry(0, 10);
        var e1 = new TestEntry(1, 100);
        pq.enqueue(e0);
        pq.enqueue(e1);
        e0.priority = 101;
        pq.update(e0, 0);
        expect(pq.dequeue()).toBe(e1);
        expect(pq.dequeue()).toBe(e0);
    });
    it('doesn\'t change index when priority doesn\'t change', function () {
        var e = new TestEntry(0, 0);
        pq.enqueue(e);
        expect(indices[0]).toEqual(0);
        pq.update(e, 0);
        expect(indices[0]).toEqual(0);
    });
    it('doesn\'t change index when priority doesn\'t trigger sift', function () {
        var e = new TestEntry(0, 0);
        pq.enqueue(e);
        expect(indices[0]).toEqual(0);
        e.priority = 1234;
        pq.update(e, 0);
        expect(indices[0]).toEqual(0);
    });
    it('changes index when priority change triggers sift', function () {
        var e = new TestEntry(0, 10);
        pq.enqueue(e);
        pq.enqueue(new TestEntry(1, 100));
        e.priority = 1000;
        pq.update(e, 0);
        expect(indices[0]).toEqual(1);
    });
});
//# sourceMappingURL=priority_queue_test.js.map