/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

// This class is based on https://github.com/mkazhdan/PoissonRecon/blob/master/Src/PoissonRecon.cpp
// Commit SHA: cc9d3ade74f6fcd944a98d9835c2fe8ad831bcc5
// The license header of the original code is shown above.

#pragma once

#include <cstddef>  // for size_t

template< typename Factory >
class PointCloudInputDataStream;

template< typename Real , unsigned int Dim , typename Index>
class MeshOutputDataWriter;

// Based on PoissonRecon from https://github.com/mkazhdan/PoissonRecon/
class PoissonReconLib {
public:
    // Input point cloud must provide the list of oriented vertices with the
    // x-, y-, and z-coordinates of the positions and the x-, y-, and
    // z-coordinates of the normals.
    class BaseCloud {
    public:
#ifdef USE_DOUBLE
        typedef double Real;
#else // !USE_DOUBLE
        typedef float  Real;
#endif // USE_DOUBLE

        virtual size_t size() const = 0;
        virtual void getPoint(size_t index, Real &x, Real &y, Real &z, Real &nx, Real &ny, Real &nz) const = 0;
    };

    // Output polygon mesh should be able to store the coordinates of the
    // vertices and the triangle indices.
    class BaseMesh {
    public:
#ifdef USE_DOUBLE
        typedef double Real;
#else // !USE_DOUBLE
        typedef float  Real;
#endif // USE_DOUBLE

#ifdef BIG_DATA
        typedef long long Index;
#else // !BIG_DATA
        typedef int       Index;
#endif // BIG_DATA

        virtual void addVertex(Real x, Real y, Real z, Real nx, Real ny, Real nz, Real d) = 0;
        virtual void addTriangle(Index i1, Index i2, Index i3) = 0;
    };

    // Reconstruct a polygon mesh from a set of oriented 3D points by solving a Poisson system
    static bool Reconstruct( const BaseCloud &cloud , BaseMesh &mesh , int depth , float finestCellWidth );
};
