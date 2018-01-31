import { Array1D, Array2D, Array3D, Array4D, NDArray } from './ndarray';
import { Rank } from './types';
export declare class Ops {
    static reverse1D(x: Array1D): Array1D;
    static reverse2D(x: Array2D, axis: number | number[]): Array2D;
    static reverse3D(x: Array3D, axis: number | number[]): Array3D;
    static reverse4D(x: Array4D, axis: number | number[]): Array4D;
    static reverse<R extends Rank>(x: NDArray<R>, axis: number | number[]): NDArray<R>;
}
