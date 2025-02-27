# distance.mojo. jpad 2025

import math
import random
import time

from collections import InlineArray
from memory import UnsafePointer


#### RGBA8 #####################################################################


@value
struct RGBA8:
    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8

    fn __init__(
        out self, r: UInt8 = 0, g: UInt8 = 0, b: UInt8 = 0, a: UInt8 = 0
    ):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    fn __str__(self) -> String:
        txt = String()
        txt += "r" + String(self.r)
        txt += "g" + String(self.g)
        txt += "b" + String(self.b)
        txt += "a" + String(self.a)
        return txt


#### Env3x3 ####################################################################


@value
struct Env3x3:
    var env: InlineArray[RGBA8, size = 3 * 3]

    fn __init__(out self, fill: RGBA8 = RGBA8(0, 0, 0, 255)):
        self.env = InlineArray[RGBA8, size = 3 * 3](fill=fill)

    fn __str__(self) -> String:
        txt = String()
        for i in range(len(self.env)):
            txt += String(self.env[i])
            if i != len(self.env) - 1:
                txt += "\n"
        return txt

    fn __getitem__(self, id: Int) -> RGBA8:
        return self.env[id]


#### Env4x4 ####################################################################


@value
struct Env4x4:
    var env: InlineArray[RGBA8, size = 4 * 4]

    fn __init__(out self, fill: RGBA8 = RGBA8(0, 0, 0, 255)):
        self.env = InlineArray[RGBA8, size = 4 * 4](fill=fill)

    fn __str__(self) -> String:
        txt = String()
        for i in range(len(self.env)):
            txt += String(self.env[i])
            if i != len(self.env) - 1:
                txt += "\n"
        return txt

    fn __getitem__(self, id: Int) -> RGBA8:
        return self.env[id]


#### utilities #################################################################


fn abs_delta(a: UInt8, b: UInt8) -> UInt8:
    if a > b:
        return a - b
    else:
        return b - a


#### distance ##################################################################


fn distance_Env3x3_L1(a: Env3x3, b: Env3x3) -> Float32:
    var d: Int = 0
    for e in range(9):
        ca = a[e]
        cb = b[e]
        d += Int(abs_delta(ca.r, cb.r))
        d += Int(abs_delta(ca.g, cb.g))
        d += Int(abs_delta(ca.b, cb.b))
        d += Int(abs_delta(ca.a, cb.a))

    # normalize (divide by 9*4*255)
    alias norm_fac = 1.0 / 9180.0
    return norm_fac * Float32(d)


fn distance_Env3x3_L1_simd(a: Env3x3, b: Env3x3) -> Float32:
    # there are 9*4 = 36 uint8 values to process: 2*16 + 4
    var d: Int = 0
    pa = UnsafePointer[Env3x3].address_of(a)
    pa_ = pa.bitcast[SIMD[DType.uint8, 16]]()
    pb = UnsafePointer[Env3x3].address_of(b)
    pb_ = pb.bitcast[SIMD[DType.uint8, 16]]()

    # 1st block
    vmin = min(pa_[0], pb_[0])
    vmax = max(pa_[0], pb_[0])
    d8_ = vmax - vmin
    d16_ = SIMD[DType.uint16, 16](d8_)  # 256-bit needed here
    d += Int(d16_.reduce_add())

    # 2nd block
    vmin = min(pa_[1], pb_[1])
    vmax = max(pa_[1], pb_[1])
    d8_ = vmax - vmin
    d16_ = SIMD[DType.uint16, 16](d8_)
    d += Int(d16_.reduce_add())

    # 3rd (partial) block
    paa = UnsafePointer[SIMD[DType.uint8, 16]].address_of(pa_[2])
    paa_ = paa.bitcast[SIMD[DType.uint8, 4]]()
    pbb = UnsafePointer[SIMD[DType.uint8, 16]].address_of(pb_[2])
    pbb_ = pbb.bitcast[SIMD[DType.uint8, 4]]()

    vminn = min(paa_[], pbb_[])
    vmaxn = max(paa_[], pbb_[])
    d8n_ = vmaxn - vminn
    d16n_ = SIMD[DType.uint16, 4](d8n_)
    d += Int(d16n_.reduce_add())

    # normalize (divide by 9*4*255)
    alias norm_fac = 1.0 / 9180.0
    return norm_fac * Float32(d)


fn distance_Env4x4_L1(a: Env4x4, b: Env4x4) -> Float32:
    var d: Int = 0
    for e in range(16):
        ca = a[e]
        cb = b[e]
        d += Int(abs_delta(ca.r, cb.r))
        d += Int(abs_delta(ca.g, cb.g))
        d += Int(abs_delta(ca.b, cb.b))
        d += Int(abs_delta(ca.a, cb.a))

    # normalize (divide by 16*4*255)
    alias norm_fac = 1.0 / 16320.0
    return norm_fac * Float32(d)


fn distance_Env4x4_L1_simd(a: Env4x4, b: Env4x4) -> Float32:
    var d: Int32 = 0
    pa = UnsafePointer[Env4x4].address_of(a)
    pa_ = pa.bitcast[SIMD[DType.uint8, 64]]()
    pb = UnsafePointer[Env4x4].address_of(b)
    pb_ = pb.bitcast[SIMD[DType.uint8, 64]]()

    vmin = min(pa_[0], pb_[])
    vmax = max(pa_[0], pb_[])
    d8_ = vmax - vmin
    d16_ = SIMD[DType.uint16, 64](d8_)
    d += Int(d16_.reduce_add())

    # normalize (divide by 16*4*255)
    alias norm_fac = 1.0 / 16320.0
    return norm_fac * Float32(d)


#### benchmarks ################################################################


fn time_distance_Env3x3_L1() -> Float32:
    var r: Float32 = 0.0
    var a = Env3x3(RGBA8(0, 0, 0, 0))
    var b = Env3x3(RGBA8(255, 255, 255, 255))

    t0 = time.perf_counter()
    for i in range(1000000):
        r += distance_Env3x3_L1(a, b)

    t1 = time.perf_counter()
    dt = t1 - t0
    print(r)

    return dt.cast[DType.float32]()


fn time_distance_Env3x3_L1_simd() -> Float32:
    var r: Float32 = 0.0
    var a = Env3x3(RGBA8(0, 0, 0, 0))
    var b = Env3x3(RGBA8(255, 255, 255, 255))

    t0 = time.perf_counter()
    for i in range(1000000):
        r += distance_Env3x3_L1_simd(a, b)

    t1 = time.perf_counter()
    dt = t1 - t0
    print(r)

    return dt.cast[DType.float32]()


fn time_distance_Env4x4_L1() -> Float32:
    var r: Float32 = 0.0
    var a = Env4x4(RGBA8(0, 0, 0, 0))
    var b = Env4x4(RGBA8(255, 255, 255, 255))

    t0 = time.perf_counter()
    for i in range(1000000):
        r += distance_Env4x4_L1(a, b)

    t1 = time.perf_counter()
    dt = t1 - t0
    print(r)

    return dt.cast[DType.float32]()


fn time_distance_Env4x4_L1_simd() -> Float32:
    var r: Float32 = 0.0
    var a = Env4x4(RGBA8(0, 0, 0, 0))
    var b = Env4x4(RGBA8(255, 255, 255, 255))

    t0 = time.perf_counter()
    for i in range(1000000):
        r += distance_Env4x4_L1_simd(a, b)

    t1 = time.perf_counter()
    dt = t1 - t0
    print(r)

    return dt.cast[DType.float32]()


#### high level ################################################################


def main():
    print("distance computation experiments...")
    random.seed()  # seed the random number generator with the current time

    # distance between two Env3x3 instances
    e3a = Env3x3(RGBA8(0, 0, 0, 0))
    e3b = Env3x3(RGBA8(255, 255, 255, 255))

    e3d = distance_Env3x3_L1(e3a, e3b)
    print("Env3x3 L1 distance       ", e3d)
    e3d = distance_Env3x3_L1_simd(e3a, e3b)
    print("Env3x3 L1 distance (simd)", e3d)

    # distance between two Env4x4 instances
    e4a = Env4x4(RGBA8(0, 0, 0, 0))
    e4b = Env4x4(RGBA8(255, 255, 255, 255))

    e4d = distance_Env4x4_L1(e4a, e4b)
    print("Env4x4 L1 distance       ", e4d)
    e4d = distance_Env4x4_L1_simd(e4a, e4b)
    print("Env4x4 L1 distance (simd)", e4d)

    #### benchmarks ############################################################

    print("distance_Env3x3_L1()      dt", time_distance_Env3x3_L1())
    print("distance_Env3x3_L1()_simd dt", time_distance_Env3x3_L1_simd())

    print("distance_Env4x4_L1()      dt", time_distance_Env4x4_L1())
    print("distance_Env4x4_L1_simd() dt", time_distance_Env4x4_L1_simd())
