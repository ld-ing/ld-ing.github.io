import { Array1D, Array2D, Array3D, NDArray } from './ndarray';
import { RandNormalDataTypes } from './rand';
import { DataType, Rank, ShapeMap } from './types';
export declare class Ops {
    static ones<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): NDArray<R>;
    static zeros<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): NDArray<R>;
    static onesLike<T extends NDArray>(x: T): T;
    static zerosLike<T extends NDArray>(x: T): T;
    static clone<T extends NDArray>(x: T): T;
    static randNormal<R extends Rank>(shape: ShapeMap[R], mean?: number, stdDev?: number, dtype?: keyof RandNormalDataTypes, seed?: number): NDArray<R>;
    static truncatedNormal<R extends Rank>(shape: ShapeMap[R], mean?: number, stdDev?: number, dtype?: keyof RandNormalDataTypes, seed?: number): NDArray<R>;
    static randUniform<R extends Rank>(shape: ShapeMap[R], a: number, b: number, dtype?: DataType): NDArray<R>;
    static rand<R extends Rank>(shape: ShapeMap[R], randFunction: () => number, dtype?: DataType): NDArray<R>;
    static multinomial(probabilities: Array1D | Array2D, numSamples: number, seed?: number): Array1D | Array2D;
    static oneHot(indices: Array1D, depth: number, onValue?: number, offValue?: number): Array2D;
    static fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels?: number): Array3D;
    static reshape<R2 extends Rank>(x: NDArray, newShape: ShapeMap[R2]): NDArray<R2>;
    static cast<T extends NDArray>(x: T, newDType: DataType): T;
    static tile<T extends NDArray>(x: T, reps: number[]): T;
    static gather<T extends NDArray>(x: T, indices: Array1D, axis?: number): T;
    static pad1D(x: Array1D, paddings: [number, number], constantValue?: number): Array1D;
    static pad2D(x: Array2D, paddings: [[number, number], [number, number]], constantValue?: number): Array2D;
}
