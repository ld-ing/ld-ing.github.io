import { NDArray } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface ReshapeNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: NDArray;
        };
        args: {
            newShape: number[];
        };
    };
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        x: () => NDArray;
    };
}
