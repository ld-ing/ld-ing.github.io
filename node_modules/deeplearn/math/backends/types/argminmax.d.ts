import { NDArray } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface ArgMaxNode extends KernelNode {
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
export interface ArgMinNode extends KernelNode {
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
