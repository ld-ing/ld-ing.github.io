import { Array1D, NDArray } from '../../ndarray';
import { Rank } from '../../types';
import { KernelNode } from '../tape_types';
export interface GatherNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
            indices: Array1D;
        };
        args: {
            axis: number;
        };
    };
    output: T;
    gradient: (dy: NDArray<R>, y: T) => {
        x: () => NDArray<R>;
        indices: () => Array1D;
    };
}
