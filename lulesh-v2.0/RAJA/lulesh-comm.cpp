#include "lulesh.hpp"

// If no MPI, then this whole file is stubbed out
#if USE_MPI

#include <mpi.h>
#include <string.h>

/* Comm Routines */

#define ALLOW_UNPACKED_PLANE false
#define ALLOW_UNPACKED_ROW   false
#define ALLOW_UNPACKED_COL   false

/*
   There are coherence issues for packing and unpacking message
   buffers.  Ideally, you would like a lot of threads to 
   cooperate in the assembly/dissassembly of each message.
   To do that, each thread should really be operating in a
   different coherence zone.

   Let's assume we have three fields, f1 through f3, defined on
   a 61x61x61 cube.  If we want to send the block boundary
   information for each field to each neighbor processor across
   each cube face, then we have three cases for the
   memory layout/coherence of data on each of the six cube
   boundaries:

      (a) Two of the faces will be in contiguous memory blocks
      (b) Two of the faces will be comprised of pencils of
          contiguous memory.
      (c) Two of the faces will have large strides between
          every value living on the face.

   How do you pack and unpack this data in buffers to
   simultaneous achieve the best memory efficiency and
   the most thread independence?

   Do do you pack field f1 through f3 tighly to reduce message
   size?  Do you align each field on a cache coherence boundary
   within the message so that threads can pack and unpack each
   field independently?  For case (b), do you align each
   boundary pencil of each field separately?  This increases
   the message size, but could improve cache coherence so
   each pencil could be processed independently by a separate
   thread with no conflicts.

   Also, memory access for case (c) would best be done without
   going through the cache (the stride is so large it just causes
   a lot of useless cache evictions).  Is it worth creating
   a special case version of the packing algorithm that uses
   non-coherent load/store opcodes?
*/

/******************************************/


/* doRecv flag only works with regular block structure */
void CommRecv(Domain& domain, int msgType, Index_t xferFields0,
              Index_t dx, Index_t dy, Index_t dz, int myRank) {

   dx = 11;
   dy = 11;
   dx = 11;
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   
   Index_t xferFields = 6 ; /* x, y, z, xd, yd, zd */
   Domain_member fieldData[6] ;
   MPI_Status status ;

   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::x ;
   fieldData[2] = &Domain::x ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::xd ;
   fieldData[5] = &Domain::xd ;

   Index_t opCount = dy * dz ;

   if (myRank == 0) {
      int fromRank = 1 - myRank ;
      int recvCount = dy * dz * xferFields ;
      MPI_Recv(&domain.commDataRecv[0],
                recvCount, baseType, fromRank, msgType,
                MPI_COMM_WORLD, &status) ;
         Real_t *srcAddr = &domain.commDataRecv[0];
	 for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(dx - 1 + i*dx*dy + j*dx) = srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
      }
}

/******************************************/
static inline
void CommSend(Domain& domain, int msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz, int myRank)
{
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;

      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dy * dz ;

      if (myRank == 1) {
         Real_t *destAddr = &domain.commDataSend[0];
	 for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  destAddr[i*dy + j] = (domain.*src)(i*dx*dy + j*dx) ;
               }
            }
            destAddr += sendCount ;
         }

         MPI_Send(&domain.commDataSend[0], xferFields*sendCount, baseType,
                   myRank - 1, msgType,
                   MPI_COMM_WORLD);
      }
}

/******************************************/

void CommSBN(Domain& domain, int xferFields, Domain_member *fieldData) {

   if (domain.numRanks() == 1)
      return ;

   /* summation order should be from smallest value to largest */
   /* or we could try out kahan summation! */
   MPI_Request req;
         MPI_Isend(nullptr, 0, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &req);

   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX() + 1 ;
   Index_t dy = domain.sizeY() + 1 ;
   Index_t dz = domain.sizeZ() + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   Index_t rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = 1 ;
   if (domain.rowLoc() == 0) {
      rowMin = 0 ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = 0 ;
   }
   if (domain.colLoc() == 0) {
      colMin = 0 ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = 0 ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = 0 ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = 0 ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(i) += srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(dx*dy*(dz - 1) + i) += srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  (domain.*dest)(i*dx*dy + j) += srcAddr[i*dx + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dx; ++j) {
                  (domain.*dest)(dx*(dy - 1) + i*dx*dy + j) += srcAddr[i*dx + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(i*dx*dy + j*dx) += srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<dz; ++i) {
               for (Index_t j=0; j<dy; ++j) {
                  (domain.*dest)(dx - 1 + i*dx*dy + j*dx) += srcAddr[i*dy + j] ;
               }
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin & colMin) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMin) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMin) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMax) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx*dy - 1 + i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMax) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*(dy-1) + dx*dy*(dz-1) + i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMax) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + dx - 1 + i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMin) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx*(dy-1) + i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMax) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMax) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx*dy*(dz-1) + i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMax) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dz; ++i) {
            (domain.*dest)(dx - 1 + i*dx*dy) += srcAddr[i] ;
         }
         srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMin) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dx; ++i) {
            (domain.*dest)(dx*(dy - 1) + i) += srcAddr[i] ;
         }
         srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMin) {
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
         for (Index_t i=0; i<dy; ++i) {
            (domain.*dest)(dx - 1 + i*dx) += srcAddr[i] ;
         }
         srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMin & planeMin) {
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(0) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin & colMin & planeMax) {
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMin) {
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMax) {
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMin) {
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMax) {
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMin) {
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMax) {
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*dz - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         (domain.*fieldData[fi])(idx) += comBuf[fi] ;
      }
      ++cmsg ;
   }
}

/******************************************/

void CommSyncPosVel(Domain& domain, Index_t dx, Index_t dy, Index_t dz, int myRank) {
}

/******************************************/

void CommMonoQ(Domain& domain)
{
   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   Index_t xferFields = 3 ; /* delv_xi, delv_eta, delv_zeta */
   Domain_member fieldData[3] ;
   Index_t fieldOffset[3] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t dx = domain.sizeX() ;
   Index_t dy = domain.sizeY() ;
   Index_t dz = domain.sizeZ() ;
   MPI_Status status ;
   Real_t *srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   /* point into ghost data area */
   // fieldData[0] = &(domain.delv_xi(domain.numElem())) ;
   // fieldData[1] = &(domain.delv_eta(domain.numElem())) ;
   // fieldData[2] = &(domain.delv_zeta(domain.numElem())) ;
   fieldData[0] = &Domain::delv_xi ;
   fieldData[1] = &Domain::delv_eta ;
   fieldData[2] = &Domain::delv_zeta ;
   fieldOffset[0] = domain.numElem() ;
   fieldOffset[1] = domain.numElem() ;
   fieldOffset[2] = domain.numElem() ;


   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            for (Index_t i=0; i<opCount; ++i) {
               (domain.*dest)(fieldOffset[fi] + i) = srcAddr[i] ;
            }
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
}

#endif
