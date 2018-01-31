import { Array1D, Array2D, Array3D, Array4D, NDArray } from './ndarray';
import { Rank, ShapeMap } from './types';
export declare class Ops {
    static slice1D(x: Array1D, begin: number, size: number): Array1D;
    static slice2D(x: Array2D, begin: [number, number], size: [number, number]): Array2D;
    static slice3D(x: Array3D, begin: [number, number, number], size: [number, number, number]): Array3D;
    static slice4D(x: Array4D, begin: [number, number, number, number], size: [number, number, number, number]): Array4D;
    static slice<R extends Rank>(x: NDArray<R>, begin: ShapeMap[R], size: ShapeMap[R]): NDArray<R>;
}
