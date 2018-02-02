/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as dl from 'deeplearn';

export function createInputAtlas(
    imageWidth: number, imageHeight: number, 
    inputNumDimensions: number, numLatentVariables: number) {
  const coords = new Float32Array(imageWidth * imageHeight 
    * inputNumDimensions);
  let dst = 0;
  for (let i = 0; i < imageWidth * imageHeight; i++) {
    for (let d = 0; d < inputNumDimensions; d++) {
      const x = i % imageWidth;
      const y = Math.floor(i / imageWidth);
      const coord = imagePixelToNormalizedCoord(
          x, y, imageWidth, imageHeight, numLatentVariables);
      coords[dst++] = coord[d];
    }
  }

  return dl.Array2D.new([imageWidth * imageHeight, 
    inputNumDimensions], coords);
}

// Normalizes x, y to -.5 <=> +.5, adds a radius term, and pads zeros with the
// number of z parameters that will get added by the add z shader.
export function imagePixelToNormalizedCoord(
    x: number, y: number, imageWidth: number, imageHeight: number,
    zSize: number): number[] {
  const halfWidth = imageWidth * 0.5;
  const halfHeight = imageHeight * 0.5;
  const normX = (x - halfWidth) / imageWidth;
  const normY = (y - halfHeight) / imageHeight;

  const r = Math.sqrt(normX * normX + normY * normY);

  const result = [normX, normY, r];

  return result;
}
