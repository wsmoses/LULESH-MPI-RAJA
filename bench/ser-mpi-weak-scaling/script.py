#!/usr/bin/python3.8
import os

def printfun(rank, blocklist,itercount):
  for s in blocklist:
      for mode in ["","-grad"]:
        os.system("mpirun -n {} taskset -c 0-{} numactl -i all ~/enzyme-sc22/LULESH-RAJA/build/bin/lulesh-v2.0-RAJA-seq-mpi{}.exe -s {} -i {} > ser-mpi{}_{}_{}_{}.txt".format(rank,rank-1,mode,s,itercount,mode,rank,itercount,s))

itercount=100
printfun(1, [96],itercount)
printfun(8, [96],itercount)
printfun(27, [96],itercount)
printfun(64, [96],itercount)
