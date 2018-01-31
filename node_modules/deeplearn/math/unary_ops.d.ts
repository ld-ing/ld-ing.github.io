import { NDArray } from './ndarray';
export declare class Ops {
    static neg<T extends NDArray>(x: T): T;
    static ceil<T extends NDArray>(x: T): T;
    static floor<T extends NDArray>(x: T): T;
    static exp<T extends NDArray>(x: T): T;
    static log<T extends NDArray>(x: T): T;
    static sqrt<T extends NDArray>(x: T): T;
    static square<T extends NDArray>(x: T): T;
    static abs<T extends NDArray>(x: T): T;
    static clip<T extends NDArray>(x: T, min: number, max: number): T;
    static relu<T extends NDArray>(x: T): T;
    static elu<T extends NDArray>(x: T): T;
    static selu<T extends NDArray>(x: T): T;
    static leakyRelu<T extends NDArray>(x: T, alpha?: number): T;
    static prelu<T extends NDArray>(x: T, alpha: T): T;
    static sigmoid<T extends NDArray>(x: T): T;
    static sin<T extends NDArray>(x: T): T;
    static cos<T extends NDArray>(x: T): T;
    static tan<T extends NDArray>(x: T): T;
    static asin<T extends NDArray>(x: T): T;
    static acos<T extends NDArray>(x: T): T;
    static atan<T extends NDArray>(x: T): T;
    static sinh<T extends NDArray>(x: T): T;
    static cosh<T extends NDArray>(x: T): T;
    static tanh<T extends NDArray>(x: T): T;
    static step<T extends NDArray>(x: T, alpha?: number): T;
}
