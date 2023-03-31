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

#include "PoissonReconLib.h"

#include "PreProcessor.h"

#undef USE_DOUBLE                               // If enabled, double-precesion is used

#define DATA_DEGREE 0                           // The order of the B-Spline used to splat in data for color interpolation
#define WEIGHT_DEGREE 2                         // The order of the B-Spline used to splat in the weights for density estimation
#define NORMAL_DEGREE 2                         // The order of the B-Spline used to splat in the normals for constructing the Laplacian constraints
#define DEFAULT_FEM_DEGREE 1                    // The default finite-element degree
#define DEFAULT_FEM_BOUNDARY BOUNDARY_NEUMANN   // The default finite-element boundary type
#define DEFAULT_DIMENSION 3                     // The dimension of the system

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "MyMiscellany.h"
#include "CmdLineParser.h"
//#include "PPolynomial.h"
#include "FEMTree.h"
//#include "Ply.h"
#include "VertexFactory.h"
//#include "Image.h"
//#include "RegularGrid.h"

namespace {

MessageWriter messageWriter;

const float DefaultPointWeightMultiplier = 2.f;

enum NormalType
{
    NORMALS_NONE ,
    NORMALS_SAMPLES ,
    NORMALS_GRADIENTS ,
    NORMALS_COUNT
};
//const char* NormalsNames[] = { "none" , "samples" , "gradients" };

//cmdLineParameter< char* >
    //In( "in" ) ,
    //Out( "out" ) ,
    //TempDir( "tempDir" ) ,
    //Grid( "grid" ) ,
    //Tree( "tree" ) ,
    //Envelope( "envelope" ) ,
    //EnvelopeGrid( "envelopeGrid" ),
    //Transform( "xForm" );

cmdLineReadable
    Performance( "performance" ) ,
    ShowResidual( "showResidual" ) ,
    //NoComments( "noComments" ) ,
    PolygonMesh( "polygonMesh" ) ,
    NonManifold( "nonManifold" ) ,
    //ASCII( "ascii" ) ,
    Density( "density" ) ,
    LinearFit( "linearFit" ) ,
    //PrimalGrid( "primalGrid" ) ,
    //ExactInterpolation( "exact" ) ,
    //Colors( "colors" ) ,
    //InCore( "inCore" ) ,
    //NoDirichletErode( "noErode" ) ,
    Verbose( "verbose" );

cmdLineParameter< int >
    //Degree( "degree" , DEFAULT_FEM_DEGREE ) ,
    Depth( "depth" , 8 ) ,
    KernelDepth( "kernelDepth" ) ,
    SolveDepth( "solveDepth" ) ,
    EnvelopeDepth( "envelopeDepth" ) ,
    Iters( "iters" , 8 ) ,
    FullDepth( "fullDepth" , 5 ) ,
    BaseDepth( "baseDepth" ) ,
    BaseVCycles( "baseVCycles" , 1 ) ,
    //BType( "bType" , DEFAULT_FEM_BOUNDARY+1 ) ,
    Normals( "normals" , NORMALS_NONE ) ,
    MaxMemoryGB( "maxMemory" , 0 ) ,
#ifdef _OPENMP
    ParallelType( "parallel" , (int)ThreadPool::OPEN_MP ) ,
#else // !_OPENMP
    ParallelType( "parallel" , (int)ThreadPool::THREAD_POOL ) ,
#endif // _OPENMP
    ScheduleType( "schedule" , (int)ThreadPool::DefaultSchedule ) ,
    ThreadChunkSize( "chunkSize" , (int)ThreadPool::DefaultChunkSize ) ,
#ifdef _OPENMP
    Threads( "threads" , (int)omp_get_num_procs() );
#else // !_OPENMP
    Threads( "threads" , (int)std::thread::hardware_concurrency() );
#endif // _OPENMP

cmdLineParameter< float >
    DataX( "data" , 32.f ) ,
    SamplesPerNode( "samplesPerNode" , 1.5f ) ,
    Scale( "scale" , 1.1f ) ,
    Width( "width" , 0.f ) ,
    Confidence( "confidence" , 0.f ) ,
    ConfidenceBias( "confidenceBias" , 0.f ) ,
    CGSolverAccuracy( "cgAccuracy" , 1e-3f ) ,
    LowDepthCutOff( "lowDepthCutOff" , 0.f ) ,
    PointWeight( "pointWeight" );

cmdLineReadable* params[] =
{
    //&Degree , &BType ,
    //&In , &Depth , &Out , &Transform ,
    //&SolveDepth ,
    //&Envelope ,
    //&EnvelopeGrid ,
    //&Width ,
    //&Scale , &Verbose , &CGSolverAccuracy , &NoComments ,
    //&KernelDepth , &SamplesPerNode , &Confidence , &NonManifold , &PolygonMesh , &ASCII , &ShowResidual ,
    //&EnvelopeDepth ,
    //&NoDirichletErode ,
    //&ConfidenceBias ,
    //&BaseDepth , &BaseVCycles ,
    //&PointWeight ,
    //&Grid , &Threads ,
    //&Tree ,
    //&Density ,
    //&FullDepth ,
    //&Iters ,
    //&DataX ,
    //&Colors ,
    //&Normals ,
    //&LinearFit ,
    //&PrimalGrid ,
    //&TempDir ,
    //&ExactInterpolation ,
    //&Performance ,
    //&MaxMemoryGB ,
    //&InCore ,
    //&ParallelType ,
    //&ScheduleType ,
    //&ThreadChunkSize ,
    //&LowDepthCutOff ,
    //NULL
};

}  // anonymous namespace

template< unsigned int Dim , class Real >
struct FEMTreeProfiler
{
    double t;

    void start( void ){ t = Time() , FEMTree< Dim , Real >::ResetLocalMemoryUsage(); }
    void print( const char* header ) const
    {
        FEMTree< Dim , Real >::MemoryUsage();
        if( header ) printf( "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
        else         printf(    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
    }
    void dumpOutput( const char* header ) const
    {
        FEMTree< Dim , Real >::MemoryUsage();
        if( header ) messageWriter( "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
        else         messageWriter(    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
    }
    void dumpOutput2( std::vector< std::string >& comments , const char* header ) const
    {
        FEMTree< Dim , Real >::MemoryUsage();
        if( header ) messageWriter( comments , "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
        else         messageWriter( comments ,    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %d (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
    }
};

template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real scaleFactor )
{
    Point< Real , Dim > center = ( max + min ) / 2;
    Real scale = max[0] - min[0];
    for( int d=1 ; d<Dim ; d++ ) scale = std::max< Real >( scale , max[d]-min[d] );
    scale *= scaleFactor;
    for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
    XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
    for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
    return sXForm * tXForm;
}
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real width , Real scaleFactor , int& depth )
{
    // Get the target resolution (along the largest dimension)
    Real resolution = ( max[0]-min[0] ) / width;
    for( int d=1 ; d<Dim ; d++ ) resolution = std::max< Real >( resolution , ( max[d]-min[d] ) / width );
    resolution *= scaleFactor;
    depth = 0;
    while( (1<<depth)<resolution ) depth++;

    Point< Real , Dim > center = ( max + min ) / 2;
    Real scale = (1<<depth) * width;

    for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
    XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
    for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
    return sXForm * tXForm;
}

template< typename Real , unsigned int Dim , typename AuxData >
using InputOrientedPointStreamInfo = typename FEMTreeInitializer< Dim , Real >::template InputPointStream< VectorTypeUnion< Real , typename VertexFactory::NormalFactory< Real , Dim >::VertexType , AuxData > >;

template< typename Real , unsigned int Dim , typename AuxData >
using InputOrientedPointStream = typename InputOrientedPointStreamInfo< Real , Dim , AuxData >::StreamType;

template< class Real , unsigned int Dim , typename AuxData >
XForm< Real , Dim+1 > GetPointXForm( InputOrientedPointStream< Real , Dim , AuxData > &stream , Real width , Real scaleFactor , int& depth )
{
    Point< Real , Dim > min , max;
    InputOrientedPointStreamInfo< Real , Dim , AuxData >::BoundingBox( stream , min , max );
    return GetBoundingBoxXForm( min , max , width , scaleFactor , depth );
}
template< class Real , unsigned int Dim , typename AuxData >
XForm< Real , Dim+1 > GetPointXForm( InputOrientedPointStream< Real , Dim , AuxData > &stream , Real scaleFactor )
{
    Point< Real , Dim > min , max;
    InputOrientedPointStreamInfo< Real , Dim , AuxData >::BoundingBox( stream , min , max );
    return GetBoundingBoxXForm( min , max , scaleFactor );
}

template< unsigned int Dim , typename Real >
struct ConstraintDual
{
    Real target , weight;
    ConstraintDual( Real t , Real w ) : target(t) , weight(w){ }
    CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p ) const { return CumulativeDerivativeValues< Real , Dim , 0 >( target*weight ); };
};
template< unsigned int Dim , typename Real >
struct SystemDual
{
    Real weight;
    SystemDual( Real w ) : weight(w){ }
    CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
    CumulativeDerivativeValues< double , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< double , Dim , 0 >& dValues ) const { return dValues * weight; };
};
template< unsigned int Dim >
struct SystemDual< Dim , double >
{
    typedef double Real;
    Real weight;
    SystemDual( Real w ) : weight(w){ }
    CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
};

template< typename Real , typename SetVertexFunction , typename InputSampleDataType , typename VertexFactory , unsigned int ... FEMSigs >
void ExtractMesh
(
    UIntPack< FEMSigs ... > ,
    FEMTree< sizeof ... ( FEMSigs ) , Real >& tree ,
    const DenseNodeData< Real , UIntPack< FEMSigs ... > >& solution ,
    Real isoValue ,
    const std::vector< typename FEMTree< sizeof ... ( FEMSigs ) , Real >::PointSample > *samples ,
    std::vector< InputSampleDataType > *sampleData ,
    const typename FEMTree< sizeof ... ( FEMSigs ) , Real >::template DensityEstimator< WEIGHT_DEGREE > *density ,
    const VertexFactory &vertexFactory ,
    const InputSampleDataType &zeroInputSampleDataType ,
    SetVertexFunction SetVertex ,
    std::vector< std::string > &comments ,
    XForm< Real , sizeof...(FEMSigs)+1 > unitCubeToModel ,
    PoissonReconLib::BaseMesh &outMesh
)
{
    static const int Dim = sizeof ... ( FEMSigs );
    typedef UIntPack< FEMSigs ... > Sigs;
    typedef typename VertexFactory::VertexType Vertex;

    static const unsigned int DataSig = FEMDegreeAndBType< DATA_DEGREE , BOUNDARY_FREE >::Signature;
    typedef typename FEMTree< Dim , Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;

    FEMTreeProfiler< Dim , Real > profiler;

    CoredMeshData< Vertex , node_index_type > *mesh = new CoredVectorMeshData< Vertex , node_index_type >();

    profiler.start();
    typename IsoSurfaceExtractor< Dim , Real , Vertex >::IsoStats isoStats;
    if( sampleData )
    {
        SparseNodeData< ProjectiveData< InputSampleDataType , Real > , IsotropicUIntPack< Dim , DataSig > > _sampleData = tree.template setExtrapolatedDataField< DataSig , false >( *samples , *sampleData , (DensityEstimator*)NULL );
        for( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* n = tree.tree().nextNode() ; n ; n=tree.tree().nextNode( n ) )
        {
            ProjectiveData< InputSampleDataType , Real >* clr = _sampleData( n );
            if( clr ) (*clr) *= (Real)pow( DataX.value , tree.depth( n ) );
        }
        isoStats = IsoSurfaceExtractor< Dim , Real , Vertex >::template Extract< InputSampleDataType >( Sigs() , UIntPack< WEIGHT_DEGREE >() , UIntPack< DataSig >() , tree , density , &_sampleData , solution , isoValue , *mesh , zeroInputSampleDataType , SetVertex , !LinearFit.set , Normals.value==NORMALS_GRADIENTS , !NonManifold.set , PolygonMesh.set , false );
    }
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
    else isoStats = IsoSurfaceExtractor< Dim , Real , Vertex >::template Extract< InputSampleDataType >( Sigs() , UIntPack< WEIGHT_DEGREE >() , UIntPack< DataSig >() , tree , density , (SparseNodeData< ProjectiveData< InputSampleDataType , Real > , IsotropicUIntPack< Dim , DataSig > > *)NULL , solution , isoValue , *mesh , zeroInputSampleDataType , SetVertex , !LinearFit.set , Normals.value==NORMALS_GRADIENTS , !NonManifold.set , PolygonMesh.set , false );
#else // !__GNUC__ || __GNUC__ >=5
    else isoStats = IsoSurfaceExtractor< Dim , Real , Vertex >::template Extract< InputSampleDataType >( Sigs() , UIntPack< WEIGHT_DEGREE >() , UIntPack< DataSig >() , tree , density , NULL , solution , isoValue , *mesh , zeroInputSampleDataType , SetVertex , !LinearFit.set , Normals.value==NORMALS_GRADIENTS , !NonManifold.set , PolygonMesh.set , false );
#endif // __GNUC__ || __GNUC__ < 4
    messageWriter( "Vertices / Polygons: %llu / %llu\n" , (unsigned long long)( mesh->outOfCoreVertexNum()+mesh->inCoreVertices.size() ) , (unsigned long long)mesh->polygonNum() );
    std::string isoStatsString = isoStats.toString() + std::string( "\n" );
    messageWriter( isoStatsString.c_str() );
    if( PolygonMesh.set ) profiler.dumpOutput2( comments , "#         Got polygons:" );
    else                  profiler.dumpOutput2( comments , "#        Got triangles:" );

    {
        typename VertexFactory::Transform unitCubeToModelTransform( unitCubeToModel );
        typedef MeshOutputDataWriter< Real , Dim , node_index_type > Writer;
        Writer *writer = new Writer( &outMesh );

        size_t nr_vertices = mesh->outOfCoreVertexNum()+mesh->inCoreVertices.size();
        size_t nr_faces = mesh->polygonNum();

        mesh->resetIterator();

        // write vertices
        if( vertexFactory.isStaticallyAllocated() )
        {
            for( size_t i=0 ; i<mesh->inCoreVertices.size() ; i++ )
            {
                typename VertexFactory::VertexType vertex = unitCubeToModelTransform( mesh->inCoreVertices[i] );
                writer->addVertex(vertex);
            }
            for( size_t i=0; i<mesh->outOfCoreVertexNum() ; i++ )
            {
                typename VertexFactory::VertexType vertex = vertexFactory();
                mesh->nextOutOfCoreVertex( vertex );
                vertex = unitCubeToModelTransform( vertex );
                writer->addVertex(vertex);
            }
        }
        else
        {
            assert(false);
        }

        // write faces
        typedef node_index_type OutputIndex;
        std::vector< CoredVertexIndex< node_index_type > > polygon;
        for( size_t i=0 ; i<nr_faces ; i++ )
        {
            mesh->nextPolygon( polygon );
            //unsigned int nr_vertices = polygon.size();
            OutputIndex *vertices = new OutputIndex[ polygon.size() ];
            for( int j=0 ; j<int(polygon.size()) ; j++ )
                if( polygon[j].inCore ) vertices[j] = (OutputIndex)polygon[j].idx;
                else                    vertices[j] = (OutputIndex)( polygon[j].idx + mesh->inCoreVertices.size() );
            writer->addTriangle(vertices[0], vertices[1], vertices[2]);
            delete[] vertices;
        }
        delete writer;
    }

    delete mesh;
}

template< class Real , typename AuxDataFactory , unsigned int ... FEMSigs >
void Execute( UIntPack< FEMSigs ... > , const AuxDataFactory &auxDataFactory , const PoissonReconLib::BaseCloud &cloud , PoissonReconLib::BaseMesh &mesh )
{
    static const int Dim = sizeof ... ( FEMSigs );
    typedef UIntPack< FEMSigs ... > Sigs;
    typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;
    typedef UIntPack< FEMDegreeAndBType< NORMAL_DEGREE , DerivativeBoundary< FEMSignature< FEMSigs >::BType , 1 >::BType >::Signature ... > NormalSigs;
    static const unsigned int DataSig = FEMDegreeAndBType< DATA_DEGREE , BOUNDARY_FREE >::Signature;
    typedef typename FEMTree< Dim , Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;
    typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 0 > InterpolationInfo;
    using namespace VertexFactory;

    // The factory for constructing an input sample
    typedef Factory< Real , PositionFactory< Real , Dim > , Factory< Real , NormalFactory< Real , Dim > , AuxDataFactory > > InputSampleFactory;

    // The factory for constructing an input sample's data
    typedef Factory< Real , NormalFactory< Real , Dim > , AuxDataFactory > InputSampleDataFactory;

    // The input point stream information: First piece of data is the normal; the remainder is the auxiliary data
    typedef InputOrientedPointStreamInfo< Real , Dim , typename AuxDataFactory::VertexType > InputPointStreamInfo;

    // The type of the input sample
    typedef typename InputPointStreamInfo::PointAndDataType InputSampleType;

    // The type of the input sample's data
    typedef typename InputPointStreamInfo::DataType InputSampleDataType;

    typedef            InputDataStream< InputSampleType >  InputPointStream;
    typedef TransformedInputDataStream< InputSampleType > XInputPointStream;

    InputSampleFactory inputSampleFactory( PositionFactory< Real , Dim >() , InputSampleDataFactory( NormalFactory< Real , Dim >() , auxDataFactory ) );
    InputSampleDataFactory inputSampleDataFactory( NormalFactory< Real , Dim >() , auxDataFactory );

    typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
    typedef typename FEMTreeInitializer< Dim , Real >::GeometryNodeType GeometryNodeType;
    std::vector< std::string > comments;
    messageWriter( comments , "*************************************************************\n" );
    messageWriter( comments , "*************************************************************\n" );
    messageWriter( comments , "** Running Screened Poisson Reconstruction (Version %s) **\n" , VERSION );
    messageWriter( comments , "*************************************************************\n" );
    messageWriter( comments , "*************************************************************\n" );
    if( !Threads.set ) messageWriter( comments , "Running with %d threads\n" , Threads.value );

    bool needNormalData = DataX.value>0 && Normals.value;
    bool needAuxData = DataX.value>0 && auxDataFactory.bufferSize();

    XForm< Real , Dim+1 > modelToUnitCube , unitCubeToModel;
    modelToUnitCube = XForm< Real , Dim+1 >::Identity();

    double startTime = Time();
    Real isoValue = 0;

    FEMTree< Dim , Real > tree( MEMORY_ALLOCATOR_BLOCK_SIZE );
    FEMTreeProfiler< Dim , Real > profiler;

    if( Depth.set && Width.value>0 )
    {
        WARN( "Both --" , Depth.name  , " and --" , Width.name , " set, ignoring --" , Width.name );
        Width.value = 0;
    }

    size_t pointCount;

    Real pointWeightSum;
    std::vector< typename FEMTree< Dim , Real >::PointSample >* samples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
    DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > geometryNodeDesignators;
    std::vector< InputSampleDataType >* sampleData = NULL;
    DensityEstimator* density = NULL;
    SparseNodeData< Point< Real , Dim > , NormalSigs >* normalInfo = NULL;
    Real targetValue = (Real)0.5;

    // Read in the samples (and color data)
    {
        profiler.start();
        InputPointStream* pointStream;
        sampleData = new std::vector< InputSampleDataType >();
        std::vector< InputSampleType > inCorePoints;

        {
            InputPointStream *_pointStream = new PointCloudInputDataStream< InputSampleFactory >( &cloud );
            InputSampleType p;
            while( _pointStream->next( p ) ) inCorePoints.push_back( p );
            delete _pointStream;

            pointStream = new MemoryInputDataStream< InputSampleType >( inCorePoints.size() , &inCorePoints[0] );
        }

        typename InputSampleFactory::Transform _modelToUnitCube( modelToUnitCube );
        auto XFormFunctor = [&]( InputSampleType &p ){ p = _modelToUnitCube( p ); };
        XInputPointStream _pointStream( XFormFunctor , *pointStream );
        if( Width.value>0 )
        {
            modelToUnitCube = GetPointXForm< Real , Dim , typename AuxDataFactory::VertexType >( _pointStream , Width.value , (Real)( Scale.value>0 ? Scale.value : 1. ) , Depth.value ) * modelToUnitCube;
            if( !SolveDepth.set ) SolveDepth.value = Depth.value;
            if( SolveDepth.value>Depth.value )
            {
                WARN( "Solution depth cannot exceed system depth: " , SolveDepth.value , " <= " , Depth.value );
                SolveDepth.value = Depth.value;
            }
            if( FullDepth.value>Depth.value )
            {
                WARN( "Full depth cannot exceed system depth: " , FullDepth.value , " <= " , Depth.value );
                FullDepth.value = Depth.value;
            }
            if( BaseDepth.value>FullDepth.value )
            {
                if( BaseDepth.set ) WARN( "Base depth must be smaller than full depth: " , BaseDepth.value , " <= " , FullDepth.value );
                BaseDepth.value = FullDepth.value;
            }
        }
        else modelToUnitCube = Scale.value>0 ? GetPointXForm< Real , Dim , typename AuxDataFactory::VertexType >( _pointStream , (Real)Scale.value ) * modelToUnitCube : modelToUnitCube;

        {
            typename InputSampleFactory::Transform _modelToUnitCube( modelToUnitCube );
            auto XFormFunctor = [&]( InputSampleType &p ){ p = _modelToUnitCube( p ); };
            XInputPointStream _pointStream( XFormFunctor , *pointStream );
            auto ProcessDataWithConfidence = [&]( const Point< Real , Dim > &p , typename InputPointStreamInfo::DataType &d )
            {
                Real l = (Real)Length( d.template get<0>() );
                if( !l || !std::isfinite( l ) ) return (Real)-1.;
                return (Real)pow( l , Confidence.value );
            };
            auto ProcessData = []( const Point< Real , Dim > &p , typename InputPointStreamInfo::DataType &d )
            {
                Real l = (Real)Length( d.template get<0>() );
                if( !l || !std::isfinite( l ) ) return (Real)-1.;
                d.template get<0>() /= l;
                return (Real)1.;
            };
            if( Confidence.value>0 ) pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( tree.spaceRoot() , _pointStream , Depth.value , *samples , *sampleData , true , tree.nodeAllocators[0] , tree.initializer() , ProcessDataWithConfidence );
            else                     pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( tree.spaceRoot() , _pointStream , Depth.value , *samples , *sampleData , true , tree.nodeAllocators[0] , tree.initializer() , ProcessData );
        }

        unitCubeToModel = modelToUnitCube.inverse();
        delete pointStream;

        messageWriter( "Input Points / Samples: %llu / %llu\n" , (unsigned long long)pointCount , (unsigned long long)samples->size() );
        profiler.dumpOutput2( comments , "# Read input into tree:" );
    }

    DenseNodeData< Real , Sigs > solution;
    {
        DenseNodeData< Real , Sigs > constraints;
        InterpolationInfo* iInfo = NULL;
        int solveDepth = Depth.value;

        tree.resetNodeIndices( 0 , std::make_tuple() );

        // Get the kernel density estimator
        {
            profiler.start();
            density = tree.template setDensityEstimator< 1 , WEIGHT_DEGREE >( *samples , KernelDepth.value , SamplesPerNode.value );
            profiler.dumpOutput2( comments , "#   Got kernel density:" );
        }

        // Transform the Hermite samples into a vector field
        {
            profiler.start();
            normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();
            std::function< bool ( InputSampleDataType , Point< Real , Dim >& ) > ConversionFunction = []( InputSampleDataType in , Point< Real , Dim > &out )
            {
                Point< Real , Dim > n = in.template get<0>();
                Real l = (Real)Length( n );
                // It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
                if( !l ) return false;
                out = n / l;
                return true;
            };
            std::function< bool ( InputSampleDataType , Point< Real , Dim >& , Real & ) > ConversionAndBiasFunction = []( InputSampleDataType in , Point< Real , Dim > &out , Real &bias )
            {
                Point< Real , Dim > n = in.template get<0>();
                Real l = (Real)Length( n );
                // It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
                if( !l ) return false;
                out = n / l;
                bias = (Real)( log( l ) * ConfidenceBias.value / log( 1<<(Dim-1) ) );
                return true;
            };
#if 1
            if( ConfidenceBias.value>0 ) *normalInfo = tree.setInterpolatedDataField( NormalSigs() , *samples , *sampleData , density , BaseDepth.value , Depth.value , (Real)LowDepthCutOff.value , pointWeightSum , ConversionAndBiasFunction );
            else                         *normalInfo = tree.setInterpolatedDataField( NormalSigs() , *samples , *sampleData , density , BaseDepth.value , Depth.value , (Real)LowDepthCutOff.value , pointWeightSum , ConversionFunction );
#else
            if( ConfidenceBias.value>0 ) *normalInfo = tree.setInterpolatedDataField( NormalSigs() , *samples , *sampleData , density , 0 , Depth.value , (Real)LowDepthCutOff.value , pointWeightSum , ConversionAndBiasFunction );
            else                         *normalInfo = tree.setInterpolatedDataField( NormalSigs() , *samples , *sampleData , density , 0 , Depth.value , (Real)LowDepthCutOff.value , pointWeightSum , ConversionFunction );
#endif
            ThreadPool::Parallel_for( 0 , normalInfo->size() , [&]( unsigned int , size_t i ){ (*normalInfo)[i] *= (Real)-1.; } );
            profiler.dumpOutput2( comments , "#     Got normal field:" );
            messageWriter( "Point weight / Estimated Measure: %g / %g\n" , pointWeightSum , pointCount*pointWeightSum );
        }

        if( !Density.set ) delete density , density = NULL;
        if( !needNormalData && !needAuxData ) delete sampleData , sampleData = NULL;

        // Add the interpolation constraints
        if( PointWeight.value>0 )
        {
            profiler.start();
            //if( ExactInterpolation.set ) iInfo = FEMTree< Dim , Real >::template       InitializeExactPointInterpolationInfo< Real , 0 > ( tree , *samples , ConstraintDual< Dim , Real >( targetValue , (Real)PointWeight.value * pointWeightSum ) , SystemDual< Dim , Real >( (Real)PointWeight.value * pointWeightSum ) , true , false );
            //else                         iInfo = FEMTree< Dim , Real >::template InitializeApproximatePointInterpolationInfo< Real , 0 > ( tree , *samples , ConstraintDual< Dim , Real >( targetValue , (Real)PointWeight.value * pointWeightSum ) , SystemDual< Dim , Real >( (Real)PointWeight.value * pointWeightSum ) , true , 1 );
            iInfo = FEMTree< Dim , Real >::template InitializeApproximatePointInterpolationInfo< Real , 0 > ( tree , *samples , ConstraintDual< Dim , Real >( targetValue , (Real)PointWeight.value * pointWeightSum ) , SystemDual< Dim , Real >( (Real)PointWeight.value * pointWeightSum ) , true , 1 );
            profiler.dumpOutput2( comments , "#Initialized point interpolation constraints:" );
        }

        // Trim the tree and prepare for multigrid
        {
            profiler.start();
            constexpr int MAX_DEGREE = NORMAL_DEGREE > Degrees::Max() ? NORMAL_DEGREE : Degrees::Max();
            typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs > hasNormalDataFunctor( *normalInfo );
            auto hasDataFunctor = [&]( const FEMTreeNode *node ){ return hasNormalDataFunctor( node ); };
            if( geometryNodeDesignators.size() ) tree.template finalizeForMultigrid< MAX_DEGREE , Degrees::Max() >( BaseDepth.value , FullDepth.value , hasDataFunctor , [&]( const FEMTreeNode *node ){ return node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() && geometryNodeDesignators[node]==GeometryNodeType::EXTERIOR; } , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , density , &geometryNodeDesignators ) );
            else                                 tree.template finalizeForMultigrid< MAX_DEGREE , Degrees::Max() >( BaseDepth.value , FullDepth.value , hasDataFunctor , []( const FEMTreeNode * ){ return false; } , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , density ) );

            profiler.dumpOutput2( comments , "#       Finalized tree:" );
        }
        // Add the FEM constraints
        {
            profiler.start();
            constraints = tree.initDenseNodeData( Sigs() );
            typename FEMIntegrator::template Constraint< Sigs , IsotropicUIntPack< Dim , 1 > , NormalSigs , IsotropicUIntPack< Dim , 0 > , Dim > F;
            unsigned int derivatives2[Dim];
            for( int d=0 ; d<Dim ; d++ ) derivatives2[d] = 0;
            typedef IsotropicUIntPack< Dim , 1 > Derivatives1;
            typedef IsotropicUIntPack< Dim , 0 > Derivatives2;
            for( int d=0 ; d<Dim ; d++ )
            {
                unsigned int derivatives1[Dim];
                for( int dd=0 ; dd<Dim ; dd++ ) derivatives1[dd] = dd==d ?  1 : 0;
                F.weights[d][ TensorDerivatives< Derivatives1 >::Index( derivatives1 ) ][ TensorDerivatives< Derivatives2 >::Index( derivatives2 ) ] = 1;
            }
            tree.addFEMConstraints( F , *normalInfo , constraints , solveDepth );
            profiler.dumpOutput2( comments , "#  Set FEM constraints:" );
        }

        // Free up the normal info
        delete normalInfo , normalInfo = NULL;

        // Add the interpolation constraints
        if( PointWeight.value>0 )
        {
            profiler.start();
            tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( iInfo ) );
            profiler.dumpOutput2( comments , "#Set point constraints:" );
        }

        messageWriter( "Leaf Nodes / Active Nodes / Ghost Nodes / Dirichlet Supported Nodes: %llu / %llu / %llu / %llu\n" , (unsigned long long)tree.leaves() , (unsigned long long)tree.nodes() , (unsigned long long)tree.ghostNodes() , (unsigned long long)tree.dirichletElements() );
        messageWriter( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );

        // Solve the linear system
        {
            profiler.start();
            typename FEMTree< Dim , Real >::SolverInfo sInfo;
            sInfo.cgDepth = 0 , sInfo.cascadic = true , sInfo.vCycles = 1 , sInfo.iters = Iters.value , sInfo.cgAccuracy = CGSolverAccuracy.value , sInfo.verbose = Verbose.set , sInfo.showResidual = ShowResidual.set , sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , sInfo.sliceBlockSize = 1;
            sInfo.baseVCycles = BaseVCycles.value;
            typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 1 > > F( { 0. , 1. } );
            solution = tree.solveSystem( Sigs() , F , constraints , SolveDepth.value , sInfo , std::make_tuple( iInfo ) );
            profiler.dumpOutput2( comments , "# Linear system solved:" );
            if( iInfo ) delete iInfo , iInfo = NULL;
        }
    }

    {
        profiler.start();
        double valueSum = 0 , weightSum = 0;
        typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &tree , solution );
        std::vector< double > valueSums( ThreadPool::NumThreads() , 0 ) , weightSums( ThreadPool::NumThreads() , 0 );
        ThreadPool::Parallel_for( 0 , samples->size() , [&]( unsigned int thread , size_t j )
        {
            ProjectiveData< Point< Real , Dim > , Real >& sample = (*samples)[j].sample;
            Real w = sample.weight;
            if( w>0 ) weightSums[thread] += w , valueSums[thread] += evaluator.values( sample.data / sample.weight , thread , (*samples)[j].node )[0] * w;
        }
        );
        for( size_t t=0 ; t<valueSums.size() ; t++ ) valueSum += valueSums[t] , weightSum += weightSums[t];
        isoValue = (Real)( valueSum / weightSum );
        if( !needNormalData && !needAuxData ) delete samples , samples = NULL;
        profiler.dumpOutput( "Got average:" );
        messageWriter( "Iso-Value: %e = %g / %g\n" , isoValue , valueSum , weightSum );
    }

    {
        if( Normals.value )
        {
            if( Density.set )
            {
                typedef Factory< Real , PositionFactory< Real , Dim > , NormalFactory< Real , Dim > , ValueFactory< Real > , AuxDataFactory > VertexFactory;
                VertexFactory vertexFactory( PositionFactory< Real , Dim >() , NormalFactory< Real , Dim >() , ValueFactory< Real >() , auxDataFactory );
                if( Normals.value==NORMALS_SAMPLES )
                {
                    auto SetVertex = []( typename VertexFactory::VertexType &v , Point< Real , Dim > p , Point< Real , Dim > g , Real w , InputSampleDataType d ){ v.template get<0>() = p , v.template get<1>() = d.template get<0>() , v.template get<2>() = w , v.template get<3>() = d.template get<1>(); };
                    ExtractMesh( UIntPack< FEMSigs ... >() , tree , solution , isoValue , samples , sampleData , density , vertexFactory , inputSampleDataFactory() , SetVertex , comments , unitCubeToModel , mesh );
                }
                //else if( Normals.value==NORMALS_GRADIENTS )
                //{
                //    auto SetVertex = []( typename VertexFactory::VertexType &v , Point< Real , Dim > p , Point< Real , Dim > g , Real w , InputSampleDataType d ){ v.template get<0>() = p , v.template get<1>() = -g/(1<<Depth.value) , v.template get<2>() = w , v.template get<3>() = d.template get<1>(); };
                //    ExtractMesh( UIntPack< FEMSigs ... >() , tree , solution , isoValue , samples , sampleData , density , vertexFactory , inputSampleDataFactory() , SetVertex , comments , unitCubeToModel , mesh );
                //}
            }
            else
            {
                typedef Factory< Real , PositionFactory< Real , Dim > , NormalFactory< Real , Dim > , AuxDataFactory > VertexFactory;
                VertexFactory vertexFactory( PositionFactory< Real , Dim >() , NormalFactory< Real , Dim >() , auxDataFactory );
                if( Normals.value==NORMALS_SAMPLES )
                {
                    auto SetVertex = []( typename VertexFactory::VertexType &v , Point< Real , Dim > p , Point< Real , Dim > g , Real w , InputSampleDataType d ){ v.template get<0>() = p                                                 , v.template get<1>() = d.template get<0>() , v.template get<2>() = d.template get<1>(); };
                    ExtractMesh( UIntPack< FEMSigs ... >() , tree , solution , isoValue , samples , sampleData , density , vertexFactory , inputSampleDataFactory() , SetVertex , comments , unitCubeToModel , mesh );
                }
                //else if( Normals.value==NORMALS_GRADIENTS )
                //{
                //    auto SetVertex = []( typename VertexFactory::VertexType &v , Point< Real , Dim > p , Point< Real , Dim > g , Real w , InputSampleDataType d ){ v.template get<0>() = p                                                 , v.template get<1>() = -g/(1<<Depth.value) , v.template get<2>() = d.template get<1>(); };
                //    ExtractMesh( UIntPack< FEMSigs ... >() , tree , solution , isoValue , samples , sampleData , density , vertexFactory , inputSampleDataFactory() , SetVertex , comments , unitCubeToModel , mesh );
                //}
            }
        }
        else
        {
            if( Density.set )
            {
                typedef Factory< Real , PositionFactory< Real , Dim > , ValueFactory< Real > , AuxDataFactory > VertexFactory;
                VertexFactory vertexFactory( PositionFactory< Real , Dim >() , ValueFactory< Real >() , auxDataFactory );
                auto SetVertex = []( typename VertexFactory::VertexType &v , Point< Real , Dim > p , Point< Real , Dim > g , Real w , InputSampleDataType d ){ v.template get<0>() = p , v.template get<1>() = w , v.template get<2>() = d.template get<1>(); };
                ExtractMesh( UIntPack< FEMSigs ... >() , tree , solution , isoValue , samples , sampleData , density , vertexFactory , inputSampleDataFactory() , SetVertex , comments , unitCubeToModel , mesh );
            }
            else
            {
                typedef Factory< Real , PositionFactory< Real , Dim > , AuxDataFactory > VertexFactory;
                VertexFactory vertexFactory( PositionFactory< Real , Dim >() , auxDataFactory );
                auto SetVertex = []( typename VertexFactory::VertexType &v , Point< Real , Dim > p , Point< Real , Dim > g , Real w , InputSampleDataType d ){ v.template get<0>() = p , v.template get<1>() = d.template get<1>(); };
                ExtractMesh( UIntPack< FEMSigs ... >() , tree , solution , isoValue , samples , sampleData , density , vertexFactory , inputSampleDataFactory() , SetVertex , comments , unitCubeToModel , mesh );
            }
        }
        if( sampleData ){ delete sampleData ; sampleData = NULL; }
    }
    if( density ) delete density , density = NULL;
    messageWriter( comments , "#          Total Solve: %9.1f (s), %9.1f (MB)\n" , Time()-startTime , FEMTree< Dim , Real >::MaxMemoryUsage() );
}


////////////////////////////////////////////////////////////////////////////////
#include "VertexStream.h"

template< typename Factory >
class PointCloudInputDataStream : public InputDataStream< typename Factory::VertexType >
{
    typedef typename Factory::VertexType Data;
    const PoissonReconLib::BaseCloud *_data;
    size_t _size;
    size_t _current;
public:
    PointCloudInputDataStream( const PoissonReconLib::BaseCloud *data );
    ~PointCloudInputDataStream( void );
    void reset( void );
    bool next( Data &d );
};

template< typename Factory >
PointCloudInputDataStream< Factory >::PointCloudInputDataStream( const PoissonReconLib::BaseCloud *data ) : _data(data) , _size(data->size()) , _current(0) {}
template< typename Factory >
PointCloudInputDataStream< Factory >::~PointCloudInputDataStream( void ){ ; }
template< typename Factory >
void PointCloudInputDataStream< Factory >::reset( void ) { _current=0; }
template< typename Factory >
bool PointCloudInputDataStream< Factory >::next( Data &d )
{
    if( _current>=_size ) return false;
    auto&& x = d.template get<0>();
    auto&& n = d.template get<1>().template get<0>();
    _data->getPoint(_current, x[0], x[1], x[2], n[0], n[1], n[2]);
    _current++;
    return true;
}

template< typename Real , unsigned int Dim , typename Index>
class MeshOutputDataWriter
{
    typedef typename VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > , VertexFactory::ValueFactory< Real > , VertexFactory::EmptyFactory< Real > >::VertexType VertexTypeA;
    typedef typename VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > , VertexFactory::EmptyFactory< Real > >::VertexType VertexTypeB;
    typedef typename VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::ValueFactory< Real > , VertexFactory::EmptyFactory< Real > >::VertexType VertexTypeC;
    typedef typename VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::EmptyFactory< Real > >::VertexType VertexTypeD;
    PoissonReconLib::BaseMesh *_data;
public:
    MeshOutputDataWriter( PoissonReconLib::BaseMesh *data );
    ~MeshOutputDataWriter( void );
    void addVertex( const VertexTypeA &v );
    void addVertex( const VertexTypeB &v );
    void addVertex( const VertexTypeC &v );
    void addVertex( const VertexTypeD &v );
    void addTriangle( Index i1 , Index i2 , Index i3 );
};

template< typename Real , unsigned int Dim , typename Index>
MeshOutputDataWriter< Real, Dim , Index >::MeshOutputDataWriter( PoissonReconLib::BaseMesh *data ) : _data(data) {}
template< typename Real , unsigned int Dim , typename Index>
MeshOutputDataWriter< Real, Dim , Index >::~MeshOutputDataWriter( void ){ ; }
template< typename Real , unsigned int Dim , typename Index>
void MeshOutputDataWriter< Real, Dim , Index >::addVertex( const VertexTypeA &v )
{
    Point< Real, Dim > x = v.template get<0>() ; Point< Real, Dim > n = v.template get<1>() ; Real d = v.template get<2>() ;
    _data->addVertex(x[0], x[1], x[2], n[0], n[1], n[2], d);
}
template< typename Real , unsigned int Dim , typename Index>
void MeshOutputDataWriter< Real, Dim , Index >::addVertex( const VertexTypeB &v )
{
    Point< Real, Dim > x = v.template get<0>() ; Point< Real, Dim > n = v.template get<1>() ;
    _data->addVertex(x[0], x[1], x[2], n[0], n[1], n[2], 0);
}
template< typename Real , unsigned int Dim , typename Index>
void MeshOutputDataWriter< Real, Dim , Index >::addVertex( const VertexTypeC &v )
{
    Point< Real, Dim > x = v.template get<0>() ; Real d = v.template get<1>() ;
    _data->addVertex(x[0], x[1], x[2], 0, 0, 0, d);
}
template< typename Real , unsigned int Dim , typename Index>
void MeshOutputDataWriter< Real, Dim , Index >::addVertex( const VertexTypeD &v )
{
    Point< Real, Dim > x = v.template get<0>() ;
    _data->addVertex(x[0], x[1], x[2], 0, 0, 0, 0);
}
template< typename Real , unsigned int Dim , typename Index>
void MeshOutputDataWriter< Real, Dim , Index >::addTriangle( Index i1, Index i2, Index i3 )
{
    _data->addTriangle( i1, i2, i3 );
}

bool PoissonReconLib::Reconstruct( const BaseCloud &cloud , BaseMesh &mesh , int depth , float finestCellWidth )
{
    Timer timer;
#ifdef USE_SEG_FAULT_HANDLER
    WARN( "using seg-fault handler" );
    StackTracer::exec = argv[0];
    signal( SIGSEGV , SignalHandler );
#endif // USE_SEG_FAULT_HANDLER
#ifdef ARRAY_DEBUG
    WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG
    //cmdLineParse( argc-1 , &argv[1] , params );
    {
        Density.set = true;
        if (depth != 0)
        {
            Depth.set = true;
            Depth.value = depth;
        }
        else if (finestCellWidth > 0)
        {
            Width.set = true;
            Width.value = finestCellWidth;
        }
    }
    if( MaxMemoryGB.value>0 ) SetPeakMemoryMB( MaxMemoryGB.value<<10 );
    ThreadPool::DefaultChunkSize = ThreadChunkSize.value;
    ThreadPool::DefaultSchedule = (ThreadPool::ScheduleType)ScheduleType.value;
    ThreadPool::Init( (ThreadPool::ParallelType)ParallelType.value , Threads.value );

    messageWriter.echoSTDOUT = Verbose.set;
    //if( !In.set )
    //{
    //    ShowUsage( argv[0] );
    //    return 0;
    //}

    if( !BaseDepth.set ) BaseDepth.value = FullDepth.value;
    if( !SolveDepth.set ) SolveDepth.value = Depth.value;

    if( BaseDepth.value>FullDepth.value )
    {
        if( BaseDepth.set ) WARN( "Base depth must be smaller than full depth: " , BaseDepth.value , " <= " , FullDepth.value );
        BaseDepth.value = FullDepth.value;
    }
    if( SolveDepth.value>Depth.value )
    {
        WARN( "Solution depth cannot exceed system depth: " , SolveDepth.value , " <= " , Depth.value );
        SolveDepth.value = Depth.value;
    }
    if( !KernelDepth.set ) KernelDepth.value = Depth.value-2;
    if( KernelDepth.value>Depth.value )
    {
        WARN( "Kernel depth should not exceed depth: " , KernelDepth.name , " <= " , KernelDepth.value );
        KernelDepth.value = Depth.value;
    }

    if( !EnvelopeDepth.set ) EnvelopeDepth.value = BaseDepth.value;
    if( EnvelopeDepth.value>Depth.value )
    {
        WARN( EnvelopeDepth.name , " can't be greater than " , Depth.name , ": " , EnvelopeDepth.value , " <= " , Depth.value );
        EnvelopeDepth.value = Depth.value;
    }
    if( EnvelopeDepth.value<BaseDepth.value )
    {
        WARN( EnvelopeDepth.name , " can't be less than " , BaseDepth.name , ": " , EnvelopeDepth.value , " >= " , BaseDepth.value );
        EnvelopeDepth.value = BaseDepth.value;
    }

#ifdef USE_DOUBLE
    typedef double Real;
#else // !USE_DOUBLE
    typedef float  Real;
#endif // USE_DOUBLE

    static const int Degree = DEFAULT_FEM_DEGREE;
    static const BoundaryType BType = DEFAULT_FEM_BOUNDARY;
    typedef IsotropicUIntPack< DEFAULT_DIMENSION , FEMDegreeAndBType< Degree , BType >::Signature > FEMSigs;
    //WARN( "Compiled for degree-" , Degree , ", boundary-" , BoundaryNames[ BType ] , ", " , sizeof(Real)==4 ? "single" : "double" , "-precision _only_" );
    if( !PointWeight.set ) PointWeight.value = DefaultPointWeightMultiplier*Degree;
    {
        Execute< Real >( FEMSigs() , VertexFactory::EmptyFactory< Real >() , cloud , mesh );
    }

    if( Performance.set )
    {
        printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
        printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );
    }

    ThreadPool::Terminate();
    return true;
}
