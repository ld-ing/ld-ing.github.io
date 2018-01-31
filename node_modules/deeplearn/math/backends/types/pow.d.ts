import { NDArray } from '../../ndarray';
import { Rank } from '../../types';
import { KernelNode } from '../tape_types';
export interface PowNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            base: T;
            exp: NDArray;
        };
    };
    output: T;
    gradient: (dy: NDArray<R>, y: T) => {
        base: () => NDArray<R>;
        exp: () => NDArray;
    };
}
