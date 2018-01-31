import { Array3D, Array4D } from './ndarray';
export declare class Ops {
    static resizeBilinear<T extends Array3D | Array4D>(images: T, size: [number, number], alignCorners?: boolean): T;
}
