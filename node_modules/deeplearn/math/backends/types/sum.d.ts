import { NDArray } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface SumNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: NDArray;
        };
        args: {
            axes: number[];
        };
    };
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        x: () => NDArray;
    };
}
