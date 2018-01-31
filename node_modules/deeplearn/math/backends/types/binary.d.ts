import { NDArray } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface BinaryNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            a: NDArray;
            b: NDArray;
        };
    };
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        a: () => NDArray;
        b: () => NDArray;
    };
}
