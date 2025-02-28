# mojo-25.1-flaw-1
repro case of a suspected Mojo 25.1 compiler flaw

Have a look at the benchmark timings of distance_Env3x3_L1() versus distance_Env4x4_L1(). The latter is incredibly slow compared to the former, while it it is in fact quite similar. About 67 times slower...

The SIMD versions of both functions run at similar speeds: this is what one would expect.
