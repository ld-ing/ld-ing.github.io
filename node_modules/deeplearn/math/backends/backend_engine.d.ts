import { NamedArrayMap } from '../../math/types';
import { NDArray, Scalar, Variable } from '../ndarray';
import { Rank } from '../types';
import { MathBackend } from './backend';
import { KernelConfigRegistry } from './kernel_registry';
import { TapeNodeInputGradientArrays } from './tape_types';
import { ScopeResult, ScopeResultImmediate } from './tape_util';
export declare class BackendEngine {
    private backend;
    private safeMode;
    private nextTapeNodeId;
    private activeTape;
    private gradientScopeCount;
    private customGradientDepth;
    private activeScope;
    private scopeStack;
    private debugMode;
    constructor(backend: MathBackend, safeMode: boolean);
    enableDebugMode(): void;
    executeKernel<R extends Rank, K extends keyof KernelConfigRegistry<R>, C extends KernelConfigRegistry<R>[K]['inputAndArgs']>(kernelName: K, config: C, grad?: KernelConfigRegistry<R>[K]['gradient']): KernelConfigRegistry<R>[K]['output'];
    customGradient<R extends Rank, T extends NDArray<R>>(f: () => {
        value: T;
        gradients: (dy: T, y: T) => TapeNodeInputGradientArrays;
    }, inputs: NamedArrayMap, name: string): T;
    gradients(f: () => Scalar, xs: NDArray[], returnValue: boolean): NDArray[] | {
        value: Scalar;
        gradients: NDArray[];
    };
    vjp<R extends Rank, T extends NDArray<R>>(f: () => T, xs: NDArray[], dy: T): NDArray[];
    variableGradientsAndValue(f: () => Scalar, varList: Variable[]): {
        value: Scalar;
        gradients: NamedArrayMap;
    };
    private gradientWrt<R, T>(y, xs, dy?);
    scope<T extends ScopeResult>(name: string, scopeFn: (keep: <T1 extends NDArray>(ndarray: T1) => T1, track: <T2 extends NDArray>(ndarray: T2) => T2) => T, gradientsMode: boolean): T;
    startScope(gradientsMode: boolean): void;
    endScope(result: ScopeResultImmediate, gradientsMode: boolean): void;
    keep<T extends NDArray>(result: T): T;
    track<T extends NDArray>(result: T): T;
    getBackend(): MathBackend;
}
