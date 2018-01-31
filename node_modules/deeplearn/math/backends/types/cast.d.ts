import { NDArray } from '../../ndarray';
import { DataType } from '../../types';
import { KernelNode } from '../tape_types';
export interface CastNode extends KernelNode {
    inputAndArgs: {
        inputs: {
            x: NDArray;
        };
        args: {
            newDType: DataType;
        };
    };
    output: NDArray;
    gradient: (dy: NDArray, y: NDArray) => {
        x: () => NDArray;
    };
}
