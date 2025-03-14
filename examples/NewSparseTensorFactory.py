import array
import ctypes
import os
import sys
from enum import Enum
import functools

import numpy as np

class CtypesArrayInterfaceWrapper:
    def __init__(self, array, shape=None, strides=None):
        self._array = array
        self._shape = shape
        self._strides = strides

    @property
    def __array_interface__(self):
        elem_size = ctypes.sizeof(self._array._type_)

        if self._shape is not None:
            shape = self._shape
        else:
            shape = [len(self._array)]

        if self._strides is not None:
            strides = self._strides
        else:
            stride = elem_size
            strides = []
            for dim_size in shape:
                strides.append(stride)
                stride *= dim_size

        assert len(shape) == len(strides)
           
        return {
            'version': 3,
            'shape': tuple(shape),
            'strides': tuple(strides),
            'data':  (ctypes.cast(self._array, ctypes.c_void_p).value, False),
            'typestr': '|V{}'.format(elem_size)
        }

# Mapping from ctypes to numpy types
ctypes_to_numpy = {
    ctypes.c_bool: np.bool_,
    ctypes.c_byte: np.int8,
    ctypes.c_ubyte: np.uint8,
    ctypes.c_short: np.int16,
    ctypes.c_ushort: np.uint16,
    ctypes.c_int: np.int32,
    ctypes.c_uint: np.uint32,
    ctypes.c_long: np.int64,
    ctypes.c_ulong: np.uint64,
    ctypes.c_float: np.float32,
    ctypes.c_double: np.float64,
}

numpy_to_ctypes = {
    np.bool_: ctypes.c_bool,
    np.int8: ctypes.c_byte,
    np.uint8: ctypes.c_ubyte,
    np.int16: ctypes.c_short,
    np.uint16: ctypes.c_ushort,
    np.int32: ctypes.c_int,
    np.uint32: ctypes.c_uint,
    np.int64: ctypes.c_long,
    np.uint64: ctypes.c_ulong,
    np.float16: ctypes.c_ushort,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.complex64: ctypes.c_float * 2,
    np.complex128: ctypes.c_double * 2,
}

class OverheadType(Enum):
    kIndex = 0
    kU64 = 1
    kU32 = 2
    kU16 = 3
    kU8 = 4

# Map both signed and unsigned ctypes/numpy types
# to the unsigned sparse runtime enums.
# We never expect positions or coordinates to contain
# negative values, so it is safe to reinterpret_cast from
# signed to unsigned type of the same width.
ctypes_to_overhead_type = {
    ctypes.c_ulong: OverheadType.kU64,
    ctypes.c_uint: OverheadType.kU32,
    ctypes.c_ushort: OverheadType.kU16,
    ctypes.c_ubyte: OverheadType.kU8,
    ctypes.c_long: OverheadType.kU64,
    ctypes.c_int: OverheadType.kU32,
    ctypes.c_short: OverheadType.kU16,
    ctypes.c_byte: OverheadType.kU8,
}

numpy_to_overhead_type = {
    np.uint64: OverheadType.kU64,
    np.uint32: OverheadType.kU32,
    np.uint16: OverheadType.kU16,
    np.uint8: OverheadType.kU8,
    np.int64: OverheadType.kU64,
    np.int32: OverheadType.kU32,
    np.int16: OverheadType.kU16,
    np.int8: OverheadType.kU8,
}

class PrimaryType(Enum):
    kF64 = 1
    kF32 = 2
    kF16 = 3
    kBF16 = 4
    kI64 = 5
    kI32 = 6
    kI16 = 7
    kI8 = 8
    kC64 = 9
    kC32 = 10

ctypes_to_primary_type = {
    ctypes.c_long: PrimaryType.kI64,
    ctypes.c_int: PrimaryType.kI32,
    ctypes.c_short: PrimaryType.kI16,
    ctypes.c_byte: PrimaryType.kI8,
    ctypes.c_float: PrimaryType.kF32,
    ctypes.c_double: PrimaryType.kF64,
    ctypes.c_float*2: PrimaryType.kF32,
    ctypes.c_double*2: PrimaryType.kF64,
}

numpy_to_primary_type = {
    np.int64: PrimaryType.kI64,
    np.int32: PrimaryType.kI32,
    np.int16: PrimaryType.kI16,
    np.int8: PrimaryType.kI8,
    np.float16: PrimaryType.kF16,
    np.float32: PrimaryType.kF32,
    np.float64: PrimaryType.kF64,
    np.complex64: PrimaryType.kC32,
    np.complex128: PrimaryType.kC64,
}

class LevelFormat(Enum):
    Undef = 0x00000000
    Dense = 0x00010000
    Batch = 0x00020000
    Compressed = 0x00040000
    Singleton = 0x00080000
    LooseCompressed = 0x00100000
    NOutOfM = 0x00200000

class Action(Enum):
    kEmpty = 0
    kFromReader = 1
    kPack = 2
    kSortCOOInPlace = 3

class LevelPropNonDefault(Enum):
    Nonunique = 0x0001   # 0b001
    Nonordered = 0x0002  # 0b010
    SoA = 0x0004         # 0b100

class LevelType(ctypes.Structure):
    _fields_ = [
        ("lvlBits", ctypes.c_int64)
    ]

    @classmethod
    def get(cls, lf, n, m, *properties):
        assert isinstance(lf, LevelFormat)
        assert (n & 0xFF) == n
        assert (m & 0xFF) == m
        lvlbits = lf.value
        lvlbits |= (n & 0xFF) << 32
        lvlbits |= (m & 0xFF) << 40
        for prop in properties:
            assert (prop & 0xFF) == prop
            lvlbits |= (prop & 0xFFFFFFFF)

        ret = cls()
        ret.lvlBits = lvlbits

        return ret

    def __int__(self):
        return self.lvlBits

    @classmethod
    def get_array(cls, data):
        ret = (cls * len(data))()
        print(f"{ret=}")
        for it, props in enumerate(data):
            ret[it] = cls.get(*props)
        return ret

np_levelType = np.int64

IndexType = ctypes.c_int64
np_IndexType = np.int64

# Makes a StridedMemRefType descriptor type with static rank.
@functools.cache
def make_strided_memref_type(dtype, rank: int):
    if rank > 0:
        class StridedMemRefType(ctypes.Structure):
            """Builds an empty descriptor for the given dtype, where rank>0."""

            _fields_ = [
                ("basePtr", ctypes.POINTER(dtype)),
                ("data", ctypes.POINTER(dtype)),
                ("offset", ctypes.c_int64),
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]

            def __repr__(self):
                basePtr = ctypes.addressof(self.basePtr.contents)
                data = ctypes.addressof(self.data.contents)
                offset = self.offset
                sizes = np.array(CtypesArrayInterfaceWrapper(self.sizes))
                strides = np.array(CtypesArrayInterfaceWrapper(self.strides))
                return f"StridedMemRefType<{dtype=},{rank=}>({basePtr=},{data=},{offset=},{sizes=},{strides=})"

            def __array_interface__(self):
                elem_size = ctypes.sizeof(self.basePtr.contents)

                shape = tuple(self.sizes)
                strides = tuple(x * elem_size for x in self.strides)

                assert len(shape) == len(strides)
                   
                return {
                    'version': 3,
                    'shape': tuple(shape),
                    'strides': tuple(strides),
                    'data':  (ctypes.cast(self.basePtr, ctypes.c_void_p).value, False),
                    'typestr': '|V{}'.format(elem_size)
                }

            def __str__(self):
                print(self.__array_interface__())
                print(np.asarray(self))
                return str(np.asarray(self))
                
    else:
        class StridedMemRefType(ctypes.Structure):
            """Builds an empty descriptor for the given dtype, where rank=0."""

            _fields_ = [
                ("basePtr", ctypes.POINTER(dtype)),
                ("data", ctypes.POINTER(dtype)),
                ("offset", ctypes.c_int64)
            ]

            def __repr__(self):
                basePtr = ctypes.addressof(self.basePtr.contents)
                data = ctypes.addressof(self.data.contents)
                offset = self.offset
                return f"StridedMemRefType<{dtype=},{rank=}>({basePtr=},{data=},{offset=})"

            def __array_interface__(self):
                elem_size = ctypes.sizeof(self.basePtr.contents)
                   
                return {
                    'version': 3,
                    'shape': tuple(),
                    'strides': tuple(),
                    'data':  (ctypes.cast(self.basePtr, ctypes.c_void_p).value, False),
                    'typestr': '|V{}'.format(elem_size)
                }

            def __str__(self):
                print(self.__array_interface__())
                print(np.asarray(self))
                return str(np.asarray(self))

    return StridedMemRefType

def make_strided_memref(array, dtype=None, rank=None):
    orig_dtype = dtype
    if isinstance(array, (list, tuple)):
        array = np.array(array, dtype=dtype)
        rank = 1

    ai = array.__array_interface__

    if dtype is None:
        dtype = array.dtype
    if rank is None:
        rank = len(ai['shape'])

    if isinstance(dtype, np.dtype):
        elem_size = dtype.itemsize
        dtype = numpy_to_ctypes[dtype.type]
    else:
        elem_size = ctypes.sizeof(dtype)

    base_ptr = ai['data'][0]

    cls = make_strided_memref_type(dtype, rank)

    ret = cls()
    ret.basePtr = ctypes.cast(base_ptr, ctypes.POINTER(dtype))
    ret.data = ctypes.cast(base_ptr, ctypes.POINTER(dtype))
    ret.offset = 0
    if rank > 0:
        ret.sizes = (ctypes.c_int64 * rank)(*ai['shape'])
        if ai['strides'] is not None:
            strides = ai['strides']
        else:
            strides = []
            stride = elem_size
            for s in ai['shape']:
                strides.append(stride)
                stride *= s
        ret.strides = (ctypes.c_int64 * rank)(*[x // elem_size for x in strides])

    ret._array = array
    return ret

class newSparseTensorFactory:
    def __init__(self):
        self.lib = ctypes.CDLL(os.getenv("SUPPORT_LIB"))
        self.func = self.lib._mlir_ciface_newSparseTensor
        self.func.argtypes = [
            ctypes.c_void_p, #dimSizesRef
            ctypes.c_void_p, #lvlSizesRef
            ctypes.c_void_p, #lvlTypesRef
            ctypes.c_void_p, #dim2lvlRef
            ctypes.c_void_p, #lvl2dimRef
            ctypes.c_int32, # posTp
            ctypes.c_int32, # crdTp
            ctypes.c_int32, # valTp
            ctypes.c_int32, # action
            ctypes.c_void_p # ptr
        ]
        self.func.restype = ctypes.c_void_p

        self._endLexInsert = self.lib.endLexInsert
        self._endLexInsert.argtypes = [ctypes.c_void_p]
        #self._endLexInsert.restype = ctypes.c_void

    # dtype: scalar data type of tensor
    # postype: position type (defaults to Index)
    # crdtype: coordinate type (defaults to Index)
    def __call__(self, shape, dtype, buffers=None, levelFormats=None, postype=None, crdtype=None): #dimSizes, lvlSizes, lvlTypes, dim2lvl, lvl2dim):
        rank = len(shape)
        dimSizes = make_strided_memref(list(shape), IndexType)
        lvlSizes = make_strided_memref(list(shape), IndexType)
        if levelFormats is None:
            # If levelTypes not provided, assume all levels are compressed.
            levelTypes = LevelType.get_array([(LevelFormat.Compressed, 0, 0) for x in range(rank)])
        else:
            levelTypes = LevelType.get_array([(lf, 0, 0) for lf in levelFormats])

        lvlTypes = make_strided_memref(CtypesArrayInterfaceWrapper(levelTypes), LevelType)
        dim2lvl = make_strided_memref(list(range(rank)), IndexType)
        lvl2dim = make_strided_memref(list(range(rank)), IndexType)

        # Convert data, position, and coordinate types to data/overhead type enums
        # Logic handles types in 3 formats (using float as example):
        # - ctypes.c_float
        # - numpy.float32
        # - numpy.dtype(numpy.float32)

        # Unpack numpy.dtype into the actual type, if that's what we got
        if isinstance(dtype, np.dtype):
            dtype = dtype.type
        if isinstance(postype, np.dtype):
            postype = postype.type
        if isinstance(crdtype, np.dtype):
            crdtype = crdtype.type

        if dtype.__module__ == 'numpy':
            priType = numpy_to_primary_type[dtype]
        elif dtype.__module__ == 'ctypes':
            priType = ctypes_to_primary_type[dtype]
        else:
            raise Exception("dtype = ", dtype, " must be numpy/ctypes type")

        if postype is None:
            postype = OverheadType.kIndex
        elif postype.__module__ == 'numpy':
            postype = numpy_to_overhead_type[postype]
        elif postype.__module__ == 'ctypes':
            postype = ctypes_to_overhead_type[postype]
        else:
            raise Exception("postype = ", postype, " must be None, or numpy/ctypes integer type")

        if crdtype is None:
            crdtype = OverheadType.kIndex
        elif crdtype.__module__ == 'numpy':
            crdtype = numpy_to_overhead_type[crdtype]
        elif crdtype.__module__ == 'ctypes':
            crdtype = ctypes_to_overhead_type[crdtype]
        else:
            raise Exception("crdtype = ", crdtype, " must be None, or numpy/ctypes integer type")

        tensor_data = []
        if buffers is None:
            action = Action.kEmpty.value
            bufptr = ctypes.cast(0, ctypes.c_void_p)
        else:
            action = Action.kPack.value

            buffers = [arr.__array_interface__['data'][0] for arr in buffers]
            tensor_data.extend(buffers)

            fullbuffer = np.array(buffers, dtype=np.int64)
            tensor_data.append(fullbuffer)

            bufptr = fullbuffer.ctypes.data_as(ctypes.c_void_p)

        ret = ctypes.cast(
            self.func(
                ctypes.pointer(dimSizes),
                ctypes.pointer(lvlSizes),
                ctypes.pointer(lvlTypes),
                ctypes.pointer(dim2lvl),
                ctypes.pointer(lvl2dim),
                postype.value,
                crdtype.value,
                priType.value,
                action,
                bufptr
            ),
            ctypes.c_void_p
        )
     
        self._endLexInsert(ret) #TODO: Needed for kPack?

        del tensor_data # Ensure tensor_data arrays are not deallocated during calls

        return ret
 
if __name__ == "__main__":
    test = newSparseTensorFactory()
    result = test([10, 10], ctypes.c_double)
    print("Done", result)
