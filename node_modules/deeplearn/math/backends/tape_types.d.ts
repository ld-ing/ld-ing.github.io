import { NamedArrayMap } from '../../math/types';
import { NDArray } from '../ndarray';
import { Rank } from '../types';
import { KernelConfigRegistry } from './kernel_registry';
export declare type Tape = Array<TapeNode<TapeNodeOutput>>;
export declare type TapeNodeOutput = NDArray | NamedArrayMap;
export declare type TapeNodeType = 'kernel' | 'customGradient';
export interface TapeNode<T extends TapeNodeOutput> {
    id: number;
    type: TapeNodeType;
    name: string;
    inputAndArgs: TapeNodeInputConfig;
    output: T;
    gradient: (dy: NDArray | NamedArrayMap, y: T) => TapeNodeInputGradientArrays;
}
export interface TapeNodeInputConfig {
    inputs: NamedArrayMap;
}
export declare type TapeNodeInputGradientArrays = {
    [inputName: string]: () => NDArray;
};
export interface KernelNode extends TapeNode<NDArray> {
    kernel: keyof KernelConfigRegistry<Rank>;
    inputAndArgs: KernelInputConfig;
    output: NDArray;
}
export interface KernelInputConfig extends TapeNodeInputConfig {
    inputs: NamedArrayMap;
    args?: {
        [argName: string]: any;
    };
}
