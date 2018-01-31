import { NDArray } from '../../ndarray';
import { DataType } from '../../types';
import { KernelNode } from '../tape_types';
export interface EqualNode extends KernelNode {
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
export interface LogicalNode extends KernelNode {
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
export interface WhereNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            condition: NDArray;
            a: NDArray;
            b: NDArray;
        };
        args: {
            dtype: DataType;
        };
    };
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        condition: () => NDArray;
        a: () => NDArray;
        b: () => NDArray;
    };
}
