import { NDArray } from '../../ndarray';
import { KernelNode } from '../tape_types';
export interface MinNode extends KernelNode {
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
export interface MinimumNode extends KernelNode {
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
export interface MaxNode extends KernelNode {
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
export interface MaximumNode extends KernelNode {
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
