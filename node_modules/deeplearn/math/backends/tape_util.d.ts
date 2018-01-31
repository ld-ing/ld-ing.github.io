import { NDArray, Variable } from '../ndarray';
import { RegularArray } from '../types';
import { Tape, TapeNodeInputConfig } from './tape_types';
export declare function getFilteredNodesXToY(tape: Tape, xs: NDArray[], y: NDArray): Tape;
export declare function backpropagateGradients(arrayAccumulatedGradientMap: {
    [ndarrayId: number]: NDArray;
}, filteredTape: Tape): void;
export declare function computeVariableInputs(tape: Tape, varList: Variable[]): Variable[];
export declare type ScopeResultImmediate = void | NDArray | RegularArray<NDArray> | {
    [key: string]: NDArray;
};
export declare type ScopeResult = ScopeResultImmediate | Promise<ScopeResultImmediate>;
export declare type ScopeFn<T extends ScopeResult> = (keep: <T1 extends NDArray>(ndarray: T1) => T1, track: <T2 extends NDArray>(ndarray: T2) => T2) => T;
export declare function extractNDArraysFromScopeResult(result: ScopeResultImmediate): NDArray[];
export declare function stripUndefinedInputsFromInputConfig(config: TapeNodeInputConfig): TapeNodeInputConfig;
