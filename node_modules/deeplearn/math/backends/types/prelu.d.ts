import { NDArray } from '../../ndarray';
import { Rank } from '../../types';
import { KernelNode } from '../tape_types';
export interface PReLUNode<R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: T;
            alpha: T;
        };
    };
    output: T;
    gradient: (dy: NDArray<R>, y: T) => {
        x: () => NDArray<R>;
        alpha: () => NDArray<R>;
    };
}
