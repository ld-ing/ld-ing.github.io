import { Array1D, NDArray } from '../../ndarray';
import { Rank } from '../../types';
import { KernelNode } from '../tape_types';
export interface TopKValuesNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
        args: {
            k: number;
        };
    };
    output: Array1D;
    gradient: (dy: Array1D, y: Array1D) => {
        x: () => T;
    };
}
export interface TopKIndicesNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: NDArray;
        };
        args: {
            k: number;
        };
    };
    output: Array1D;
    gradient: (dy: Array1D, y: Array1D) => {
        x: () => NDArray;
    };
}
