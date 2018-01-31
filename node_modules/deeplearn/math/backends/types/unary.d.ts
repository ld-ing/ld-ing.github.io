import { NDArray } from '../../ndarray';
import { Rank } from '../../types';
import { KernelNode } from '../tape_types';
export interface UnaryNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
    };
    output: T;
    gradient: (dy: T, y: T) => {
        x: () => T;
    };
}
export interface LeakyReluNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
        args: {
            alpha: number;
        };
    };
    output: T;
    gradient: (dy: T, y: T) => {
        x: () => T;
    };
}
export interface StepNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
        args: {
            alpha: number;
        };
    };
    output: T;
    gradient: (dy: T, y: T) => {
        x: () => T;
    };
}
export interface ClipNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
        args: {
            min: number;
            max: number;
        };
    };
    output: T;
    gradient: (dy: T, y: T) => {
        x: () => T;
    };
}
export interface TransposeNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
        args: {
            perm: number[];
        };
    };
    output: T;
    gradient: (dy: T, y: T) => {
        x: () => T;
    };
}
export interface TileNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
        };
        args: {
            reps: number[];
        };
    };
    output: T;
    gradient: (dy: T, y: T) => {
        x: () => T;
    };
}
