#if !defined(USE_MPI)
# error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

#if USE_MPI
#include <mpi.h>

/*
   define one of these three symbols:

   SEDOV_SYNC_POS_VEL_NONE
   SEDOV_SYNC_POS_VEL_EARLY
   SEDOV_SYNC_POS_VEL_LATE
*/

#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

#include <math.h>
#include <vector>

#include "RAJA/RAJA.hpp"


//**************************************************
// Allow flexibility for arithmetic representations
//**************************************************

// Precision specification
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  // 10 bytes on x86

typedef RAJA::Index_type Index_t ; // array subscript and loop index
typedef real8  Real_t ;  // floating point representation
typedef int    Int_t ;   // integer representation

enum { VolumeError = -1, QStopError = -2 } ;

inline RAJA_HOST_DEVICE
real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline RAJA_HOST_DEVICE
real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline RAJA_HOST_DEVICE
real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline RAJA_HOST_DEVICE
real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline RAJA_HOST_DEVICE
real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline RAJA_HOST_DEVICE
real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline RAJA_HOST_DEVICE
real4  FABS(real4  arg) { return fabsf(arg) ; }
inline RAJA_HOST_DEVICE
real8  FABS(real8  arg) { return fabs(arg) ; }
inline RAJA_HOST_DEVICE
real10 FABS(real10 arg) { return fabsl(arg) ; }


// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))


//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

/*
 * The implementation of the data abstraction used for lulesh
 * resides entirely in the Domain class below.  You can change
 * grouping and interleaving of fields here to maximize data layout
 * efficiency for your underlying architecture or compiler.
 *
 * For example, fields can be implemented as STL objects or
 * raw array pointers.  As another example, individual fields
 * m_x, m_y, m_z could be budled into
 *
 *    struct { Real_t x, y, z ; } *m_coord ;
 *
 * allowing accessor functions such as
 *
 *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
 *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
 *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
 */

class Domain {

   public:

   // Constructor
   Domain(Int_t numRanks, Index_t colLoc,
          Index_t rowLoc, Index_t planeLoc,
          Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

   // Destructor
   ~Domain();

   //
   // ALLOCATION
   //

   void AllocateNodePersistent(Int_t numNode) // Node-centered
   {
      m_x.reserve(numNode);  // coordinates
      m_y.reserve(numNode);
      m_z.reserve(numNode);

      m_xd.reserve(numNode); // velocities
      m_yd.reserve(numNode);
      m_zd.reserve(numNode);

      m_xdd.reserve(numNode); // accelerations
      m_ydd.reserve(numNode);
      m_zdd.reserve(numNode);

      m_fx.reserve(numNode);  // forces
      m_fy.reserve(numNode);
      m_fz.reserve(numNode);

      m_nodalMass.reserve(numNode);  // mass
   }

   void AllocateElemPersistent(Int_t numElem) // Elem-centered
   {
      m_nodelist.reserve(8*numElem);

      // elem connectivities through face
      m_lxim.reserve(numElem);
      m_lxip.reserve(numElem);
      m_letam.reserve(numElem);
      m_letap.reserve(numElem);
      m_lzetam.reserve(numElem);
      m_lzetap.reserve(numElem);

      m_elemBC.reserve(numElem);

      m_e.reserve(numElem);
      m_p.reserve(numElem);

      m_q.reserve(numElem);
      m_ql.reserve(numElem);
      m_qq.reserve(numElem);

      m_v.reserve(numElem);

      m_volo.reserve(numElem);
      m_delv.reserve(numElem);
      m_vdov.reserve(numElem);

      m_arealg.reserve(numElem);

      m_ss.reserve(numElem);

      m_elemMass.reserve(numElem);

      m_vnew.reserve(numElem) ;
   }

   void AllocateGradients(lulesh2::MemoryPool< Real_t > &pool,
                          Int_t numElem, Int_t allElem)
   {
      (void) pool ;

      // Position gradients
      m_delx_xi.reserve(numElem) ;
      m_delx_eta.reserve(numElem) ;
      m_delx_zeta.reserve(numElem) ;

      // Velocity gradients
      m_delv_xi.reserve(allElem) ;
      m_delv_eta.reserve(allElem);
      m_delv_zeta.reserve(allElem) ;
   }

   void DeallocateGradients(lulesh2::MemoryPool< Real_t > &pool)
   {
      (void) pool ;
      m_delx_zeta.clear() ;
      m_delx_zeta.shrink_to_fit();
      m_delx_eta.clear() ;
      m_delx_eta.shrink_to_fit();
      m_delx_xi.clear() ;
      m_delx_xi.shrink_to_fit();

      m_delv_zeta.clear() ;
      m_delv_zeta.shrink_to_fit();
      m_delv_eta.clear() ;
      m_delv_eta.shrink_to_fit();
      m_delv_xi.clear() ;
      m_delv_xi.shrink_to_fit();
   }

   void AllocateStrains(lulesh2::MemoryPool< Real_t > &pool,
                        Int_t numElem)
   {
      (void) pool ;

      m_dxx.reserve(numElem) ;
      m_dyy.reserve(numElem) ;
      m_dzz.reserve(numElem) ;
   }

   void DeallocateStrains(lulesh2::MemoryPool< Real_t > &pool)
   {
      (void) pool ;

      m_dzz.clear() ;
      m_dzz.shrink_to_fit();
      m_dyy.clear() ;
      m_dyy.shrink_to_fit();
      m_dxx.clear() ;
      m_dxx.shrink_to_fit();
   }

   //
   // ACCESSORS
   //

   // Node-centered

   // Nodal coordinates
   Real_t& x(Index_t idx)    { return m_x[idx] ; }
   Real_t& y(Index_t idx)    { return m_y[idx] ; }
   Real_t& z(Index_t idx)    { return m_z[idx] ; }

   // Nodal velocities
   Real_t& xd(Index_t idx)   { return m_xd[idx] ; }
   Real_t& yd(Index_t idx)   { return m_yd[idx] ; }
   Real_t& zd(Index_t idx)   { return m_zd[idx] ; }

   // Nodal accelerations
   Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }
   Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }
   Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }

   // Nodal forces
   Real_t& fx(Index_t idx)   { return m_fx[idx] ; }
   Real_t& fy(Index_t idx)   { return m_fy[idx] ; }
   Real_t& fz(Index_t idx)   { return m_fz[idx] ; }

   // Nodal mass
   Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }

   //
   // Element-centered
   //
   Index_t*  nodelist(Index_t idx) { return &m_nodelist[Index_t(8)*idx] ; }

#if !defined(LULESH_LIST_INDEXSET)
   Index_t&  perm(Index_t idx)     { return m_perm[idx] ; }
#else
   Index_t  perm(Index_t idx)     { return idx ; }
#endif

   // elem connectivities through face
   Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }
   Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }
   Index_t&  letam(Index_t idx) { return m_letam[idx] ; }
   Index_t&  letap(Index_t idx) { return m_letap[idx] ; }
   Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }
   Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }

   // elem face symm/free-surface flag
   Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }

   // Principal strains - temporary
   Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }
   Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }
   Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }

   // New relative volume - temporary
   Real_t& vnew(Index_t idx)  { return m_vnew[idx] ; }

   // Velocity gradient - temporary
   Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }
   Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }
   Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }

   // Position gradient - temporary
   Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }
   Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }
   Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }

   // Energy
   Real_t& e(Index_t idx)          { return m_e[idx] ; }

   // Pressure
   Real_t& p(Index_t idx)          { return m_p[idx] ; }

   // Artificial viscosity
   Real_t& q(Index_t idx)          { return m_q[idx] ; }

   // Linear term for q
   Real_t& ql(Index_t idx)         { return m_ql[idx] ; }
   // Quadratic term for q
   Real_t& qq(Index_t idx)         { return m_qq[idx] ; }

   // Relative volume
   Real_t& v(Index_t idx)          { return m_v[idx] ; }
   Real_t& delv(Index_t idx)       { return m_delv[idx] ; }

   // Reference volume
   Real_t& volo(Index_t idx)       { return m_volo[idx] ; }

   // volume derivative over volume
   Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }

   // Element characteristic length
   Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }

   // Sound speed
   Real_t& ss(Index_t idx)         { return m_ss[idx] ; }

   // Element mass
   Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }

#if defined(OMP_FINE_SYNC)
   Index_t nodeElemCount(Index_t idx)
   { return m_nodeElemStart[idx+1] - m_nodeElemStart[idx] ; }

   Index_t *nodeElemCornerList(Index_t idx)
   { return &m_nodeElemCornerList[m_nodeElemStart[idx]] ; }
#endif

   // Region Centered

   Index_t&  regElemSize(Index_t idx) { return m_regElemSize[idx] ; }
   Index_t&  regNumList(Index_t idx) { return m_regNumList[idx] ; }
   Index_t*  regNumList()            { return &m_regNumList[0] ; }
   Index_t*  regElemlist(Int_t r)    { return m_regElemlist[r] ; }
   Index_t&  regElemlist(Int_t r, Index_t idx)
   { return m_regElemlist[r][idx] ; }

   // Parameters

   // Cutoffs
   Real_t u_cut() const               { return m_u_cut ; }
   Real_t e_cut() const               { return m_e_cut ; }
   Real_t p_cut() const               { return m_p_cut ; }
   Real_t q_cut() const               { return m_q_cut ; }
   Real_t v_cut() const               { return m_v_cut ; }

   // Other constants (usually are settable via input file in real codes)
   Real_t hgcoef() const              { return m_hgcoef ; }
   Real_t qstop() const               { return m_qstop ; }
   Real_t monoq_max_slope() const     { return m_monoq_max_slope ; }
   Real_t monoq_limiter_mult() const  { return m_monoq_limiter_mult ; }
   Real_t ss4o3() const               { return m_ss4o3 ; }
   Real_t qlc_monoq() const           { return m_qlc_monoq ; }
   Real_t qqc_monoq() const           { return m_qqc_monoq ; }
   Real_t qqc() const                 { return m_qqc ; }

   Real_t eosvmax() const             { return m_eosvmax ; }
   Real_t eosvmin() const             { return m_eosvmin ; }
   Real_t pmin() const                { return m_pmin ; }
   Real_t emin() const                { return m_emin ; }
   Real_t dvovmax() const             { return m_dvovmax ; }
   Real_t refdens() const             { return m_refdens ; }

   // Timestep controls, etc...
   Real_t& time()                 { return m_time ; }
   Real_t& deltatime()            { return m_deltatime ; }
   Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
   Real_t& deltatimemultub()      { return m_deltatimemultub ; }
   Real_t& stoptime()             { return m_stoptime ; }
   Real_t& dtcourant()            { return m_dtcourant ; }
   Real_t& dthydro()              { return m_dthydro ; }
   Real_t& dtmax()                { return m_dtmax ; }
   Real_t& dtfixed()              { return m_dtfixed ; }

   Int_t&  cycle()                { return m_cycle ; }
   Int_t&  numRanks()             { return m_numRanks ; }

   Index_t&  colLoc()             { return m_colLoc ; }
   Index_t&  rowLoc()             { return m_rowLoc ; }
   Index_t&  planeLoc()           { return m_planeLoc ; }
   Index_t&  tp()                 { return m_tp ; }

   Index_t&  sizeX()              { return m_sizeX ; }
   Index_t&  sizeY()              { return m_sizeY ; }
   Index_t&  sizeZ()              { return m_sizeZ ; }
   Index_t&  numReg()             { return m_numReg ; }
   Index_t&  cost()               { return m_cost ; }
   Index_t&  numElem()            { return m_numElem ; }
   Index_t&  numNode()            { return m_numNode ; }

   Index_t&  maxPlaneSize()       { return m_maxPlaneSize ; }
   Index_t&  maxEdgeSize()        { return m_maxEdgeSize ; }

   //
   // Accessors for index sets
   //
   LULESH_ISET& getNodeISet()    { return m_domNodeISet ; }
   LULESH_ISET& getElemISet()    { return m_domElemISet ; }
   LULESH_ISET& getElemRegISet() { return m_domElemRegISet ; }

   LULESH_ISET& getRegionISet(int r) { return m_domRegISet[r] ; }

   LULESH_ISET& getXSymNodeISet() { return m_domXSymNodeISet ; }
   LULESH_ISET& getYSymNodeISet() { return m_domYSymNodeISet ; }
   LULESH_ISET& getZSymNodeISet() { return m_domZSymNodeISet ; }

   //
   // MPI-Related additional data
   //

#if USE_MPI
   // Communication Work space
   Real_t *commDataSend ;
   Real_t *commDataRecv ;

   // Maximum number of block neighbors
   MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners
   MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners
#endif

  private:

   void BuildMeshTopology(Index_t edgeNodes, Index_t edgeElems);
   void BuildMeshCoordinates(Index_t nx, Index_t edgeNodes);
   void SetupThreadSupportStructures();
   void CreateMeshIndexSets();
   void CreateRegionIndexSets(Int_t nreg, Int_t balance);
   void CreateSymmetryIndexSets(Index_t edgeNodes);
   void SetupCommBuffers(Index_t edgeNodes);
   void SetupElementConnectivities(Index_t edgeElems);
   void SetupBoundaryConditions(Index_t edgeElems);

   //
   // IMPLEMENTATION
   //

   /* mesh-based index sets */
   LULESH_ISET m_domNodeISet ;
   LULESH_ISET m_domElemISet ;
   LULESH_ISET m_domElemRegISet ;

   LULESH_ISET m_domXSymNodeISet ;
   LULESH_ISET m_domYSymNodeISet ;
   LULESH_ISET m_domZSymNodeISet ;

   /* region-based index sets */
   std::vector<LULESH_ISET> m_domRegISet;

   /* Node-centered */
   std::vector<Real_t> m_x ;  /* coordinates */
   std::vector<Real_t> m_y ;
   std::vector<Real_t> m_z ;

   std::vector<Real_t> m_xd ; /* velocities */
   std::vector<Real_t> m_yd ;
   std::vector<Real_t> m_zd ;

   std::vector<Real_t> m_xdd ; /* accelerations */
   std::vector<Real_t> m_ydd ;
   std::vector<Real_t> m_zdd ;

   std::vector<Real_t> m_fx ;  /* forces */
   std::vector<Real_t> m_fy ;
   std::vector<Real_t> m_fz ;

   std::vector<Real_t> m_nodalMass ;  /* mass */

   // Element-centered

   std::vector<Index_t>  m_nodelist ;     /* elemToNode connectivity */

   std::vector<Index_t>  m_lxim ;  /* element connectivity across each face */
   std::vector<Index_t>  m_lxip ;
   std::vector<Index_t>  m_letam ;
   std::vector<Index_t>  m_letap ;
   std::vector<Index_t>  m_lzetam ;
   std::vector<Index_t>  m_lzetap ;

   std::vector<Int_t>    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   std::vector<Real_t> m_dxx ;  /* principal strains -- temporary */
   std::vector<Real_t> m_dyy ;
   std::vector<Real_t> m_dzz ;

   std::vector<Real_t> m_delv_xi ;    /* velocity gradient -- temporary */
   std::vector<Real_t> m_delv_eta ;
   std::vector<Real_t> m_delv_zeta ;

   std::vector<Real_t> m_delx_xi ;    /* coordinate gradient -- temporary */
   std::vector<Real_t> m_delx_eta ;
   std::vector<Real_t> m_delx_zeta ;

   std::vector<Real_t> m_e ;   /* energy */

   std::vector<Real_t> m_p ;   /* pressure */
   std::vector<Real_t> m_q ;   /* q */
   std::vector<Real_t> m_ql ;  /* linear term for q */
   std::vector<Real_t> m_qq ;  /* quadratic term for q */

   std::vector<Real_t> m_v ;     /* relative volume */
   std::vector<Real_t> m_volo ;  /* reference volume */
   std::vector<Real_t> m_vnew ;  /* new relative volume -- temporary */
   std::vector<Real_t> m_delv ;  /* m_vnew - m_v */
   std::vector<Real_t> m_vdov ;  /* volume derivative over volume */

   std::vector<Real_t> m_arealg ;  /* characteristic length of an element */

   std::vector<Real_t> m_ss ;      /* "sound speed" */

   std::vector<Real_t> m_elemMass ;  /* mass */

   // Cutoffs (treat as constants)
   const Real_t  m_e_cut ;             // energy tolerance
   const Real_t  m_p_cut ;             // pressure tolerance
   const Real_t  m_q_cut ;             // q tolerance
   const Real_t  m_v_cut ;             // relative volume tolerance
   const Real_t  m_u_cut ;             // velocity tolerance

   // Other constants (usually setable, but hardcoded in this proxy app)

   const Real_t  m_hgcoef ;            // hourglass control
   const Real_t  m_ss4o3 ;
   const Real_t  m_qstop ;             // excessive q indicator
   const Real_t  m_monoq_max_slope ;
   const Real_t  m_monoq_limiter_mult ;
   const Real_t  m_qlc_monoq ;         // linear term coef for q
   const Real_t  m_qqc_monoq ;         // quadratic term coef for q
   const Real_t  m_qqc ;
   const Real_t  m_eosvmax ;
   const Real_t  m_eosvmin ;
   const Real_t  m_pmin ;              // pressure floor
   const Real_t  m_emin ;              // energy floor
   const Real_t  m_dvovmax ;           // maximum allowable volume change
   const Real_t  m_refdens ;           // reference density

   // Variables to keep track of timestep, simulation time, and cycle
   Real_t  m_dtcourant ;         // courant constraint
   Real_t  m_dthydro ;           // volume change constraint
   Int_t   m_cycle ;             // iteration count for simulation
   Real_t  m_dtfixed ;           // fixed time increment
   Real_t  m_time ;              // current time
   Real_t  m_deltatime ;         // variable time increment
   Real_t  m_deltatimemultlb ;
   Real_t  m_deltatimemultub ;
   Real_t  m_dtmax ;             // maximum allowable time increment
   Real_t  m_stoptime ;          // end time for simulation

   Int_t   m_numRanks ;

   Index_t m_colLoc ;
   Index_t m_rowLoc ;
   Index_t m_planeLoc ;
   Index_t m_tp ;

   Index_t m_sizeX ;
   Index_t m_sizeY ;
   Index_t m_sizeZ ;
   Index_t m_numElem ;
   Index_t m_numNode ;

   Index_t m_maxPlaneSize ;
   Index_t m_maxEdgeSize ;

   // Region information
   Index_t    m_numReg ;
   Index_t    m_cost; //imbalance cost
   Index_t *m_regElemSize ;   // Size of region sets
   Index_t *m_regNumList ;    // Region number per domain element
   Index_t **m_regElemlist ;  // region indexset

   // Permutation to pack element-centered material subsets
   // into a contiguous range per material
   Index_t *m_perm ;

#if defined(OMP_FINE_SYNC)
   Index_t *m_nodeElemStart ;
   Index_t *m_nodeElemCornerList ;
#endif

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

} ;

typedef Real_t &(Domain::* Domain_member )(Index_t) ;

struct cmdLineOpts {
   Int_t its; // -i
   Int_t nx;  // -s
   Int_t numReg; // -r
   Int_t numFiles; // -f
   Int_t showProg; // -p
   Int_t quiet; // -q
   Int_t viz; // -v
   Int_t cost; // -c
   Int_t balance; // -b
};



// Function Prototypes

// lulesh-par
RAJA_HOST_DEVICE
Real_t CalcElemVolume( const Real_t x[8],
                       const Real_t y[8],
                       const Real_t z[8]);

// lulesh-util
void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks);

// lulesh-viz
void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// lulesh-init
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side);
