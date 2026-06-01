#!/usr/bin/env python3
#如果要使用请先说明本作者是林曦
import struct, zlib, json, os, sys, time, tempfile, shutil, io, subprocess
import multiprocessing
import ctypes

_TMPDIR = tempfile.gettempdir()
try:
    import numpy as np
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(cache=False)
    def _decrypt_numba(data, mask, step, sbox_blob):
        n = len(data)
        out = np.empty(n, dtype=np.uint8)
        cm0 = np.uint16(mask[0]); cm1 = np.uint16(mask[1]); cm2 = np.uint16(mask[2])
        cm3 = np.uint16(mask[3]); cm4 = np.uint16(mask[4]); cm5 = np.uint16(mask[5])
        cs0 = np.uint16(step[0]); cs1 = np.uint16(step[1]); cs2 = np.uint16(step[2])
        cs3 = np.uint16(step[3]); cs4 = np.uint16(step[4]); cs5 = np.uint16(step[5])
        for k in range(n):
            v = np.uint16(data[k])
            v = np.uint16(sbox_blob[np.uint16(1280) + v]) ^ cm5
            v = np.uint16(sbox_blob[np.uint16(1024) + v]) ^ cm4
            v = np.uint16(sbox_blob[np.uint16(768) + v]) ^ cm3
            v = np.uint16(sbox_blob[np.uint16(512) + v]) ^ cm2
            v = np.uint16(sbox_blob[np.uint16(256) + v]) ^ cm1
            v = np.uint16(sbox_blob[v]) ^ cm0
            out[k] = np.uint8(v)
            sv = cm0 + cs0
            cm0 = sv & np.uint16(0xFF)
            if sv >= np.uint16(256):
                cm1 = (cm1 + np.uint16(1)) & np.uint16(0xFF)
            sv = cm1 + cs1
            cm1 = sv & np.uint16(0xFF)
            if sv >= np.uint16(256):
                cm2 = (cm2 + np.uint16(1)) & np.uint16(0xFF)
            sv = cm2 + cs2
            cm2 = sv & np.uint16(0xFF)
            if sv >= np.uint16(256):
                cm3 = (cm3 + np.uint16(1)) & np.uint16(0xFF)
            sv = cm3 + cs3
            cm3 = sv & np.uint16(0xFF)
            if sv >= np.uint16(256):
                cm4 = (cm4 + np.uint16(1)) & np.uint16(0xFF)
            sv = cm4 + cs4
            cm4 = sv & np.uint16(0xFF)
            if sv >= np.uint16(256):
                cm5 = (cm5 + np.uint16(1)) & np.uint16(0xFF)
            sv = cm5 + cs5
            cm5 = sv & np.uint16(0xFF)
        return out


class NlsCipher:
    def __init__(self, seed_bytes=b"\x98\x84\x5D\x9A\x9E\x8B"):
        if len(seed_bytes) < 6:
            raise ValueError("Seed must be at least 6 bytes")
        self.s1 = int.from_bytes(seed_bytes[0:2], 'little', signed=True)
        self.s2 = int.from_bytes(seed_bytes[2:4], 'little', signed=True)
        self.s3 = int.from_bytes(seed_bytes[4:6], 'little', signed=True)
        self.mask, self.step = [], []
        self.sbox_blob, self.rsbox_blob = [], []
        self._generate_keys()
        if _HAS_NUMBA:
            self._mask_arr = np.array(self.mask, dtype=np.uint8)
            self._step_arr = np.array(self.step, dtype=np.uint8)
            self._sbox_arr = np.array(self.sbox_blob, dtype=np.uint8)

    def _prng_step(self, limit):
        v5 = ctypes.c_int32(171 * self.s1 - 30269 * int(self.s1 / 177)).value
        self.s1 = v5 if v5 >= 0 else v5 + 30269
        v7 = ctypes.c_int32(172 * self.s2 - 30307 * int(self.s2 / 176)).value
        self.s2 = v7 if v7 >= 0 else v7 + 30307
        v8 = ctypes.c_int32(170 * self.s3 - 30323 * int(self.s3 / 178)).value
        self.s3 = v8 if v8 >= 0 else v8 + 30323
        f = (self.s1 / 30269.0 + self.s2 / 30307.0 + self.s3 / 30323.0)
        f -= int(f)
        return int(f * limit)

    def _generate_keys(self):
        for _ in range(6):
            self.mask.append(self._prng_step(256) & 0xFF)
            s = self._prng_step(128)
            self.step.append((s * 2 + 1) & 0xFF)
            p_arr = list(range(256))
            sbox = [0] * 256
            rsbox = [0] * 256
            n = 256
            while n >= 2:
                idx = self._prng_step(n)
                sel, last = p_arr[idx], p_arr[n - 1]
                p_arr[idx], p_arr[n - 1] = last, sel
                sbox[sel] = n - 1
                rsbox[n - 1] = sel
                n -= 1
            sbox[p_arr[0]] = 0
            rsbox[0] = p_arr[0]
            self.sbox_blob.extend(sbox)
            self.rsbox_blob.extend(rsbox)

    def decrypt(self, data):
        if _HAS_NUMBA:
            if isinstance(data, (bytes, bytearray)):
                data_arr = np.frombuffer(data, dtype=np.uint8)
            else:
                data_arr = np.frombuffer(bytes(data), dtype=np.uint8)
            return _decrypt_numba(data_arr, self._mask_arr, self._step_arr, self._sbox_arr).tobytes()
        out = bytearray(len(data))
        cm0, cm1, cm2, cm3, cm4, cm5 = self.mask
        cs0, cs1, cs2, cs3, cs4, cs5 = self.step
        sb = self.sbox_blob
        s0, s1, s2, s3, s4, s5 = sb[0:256], sb[256:512], sb[512:768], sb[768:1024], sb[1024:1280], sb[1280:1536]
        for k in range(len(data)):
            v = data[k]
            v = s5[v] ^ cm5
            v = s4[v] ^ cm4
            v = s3[v] ^ cm3
            v = s2[v] ^ cm2
            v = s1[v] ^ cm1
            v = s0[v] ^ cm0
            out[k] = v
            sv0 = cm0 + cs0
            cm0 = sv0 & 0xFF
            if sv0 >= 256:
                cm1 = (cm1 + 1) & 0xFF
            sv1 = cm1 + cs1
            cm1 = sv1 & 0xFF
            if sv1 >= 256:
                cm2 = (cm2 + 1) & 0xFF
            sv2 = cm2 + cs2
            cm2 = sv2 & 0xFF
            if sv2 >= 256:
                cm3 = (cm3 + 1) & 0xFF
            sv3 = cm3 + cs3
            cm3 = sv3 & 0xFF
            if sv3 >= 256:
                cm4 = (cm4 + 1) & 0xFF
            sv4 = cm4 + cs4
            cm4 = sv4 & 0xFF
            if sv4 >= 256:
                cm5 = (cm5 + 1) & 0xFF
            sv5 = cm5 + cs5
            cm5 = sv5 & 0xFF
        return out

def _decrypt_data(data):
    zc = b""
    if data[0] == 0x35:
        hdr = bytearray(data[:4])
        for i in range(4):
            hdr[i] ^= b"MCPK"[i]
        zc = hdr + data[4:]
    elif data[:2] == b'\xE5\x1F':
        zc = NlsCipher().decrypt(data)
    else:
        return data
    if len(zc) > 2 and zc[0] == 0x78 and zc[1] in (0x01, 0x9C, 0xDA):
        try:
            fc = zlib.decompress(zc)
            if data[:2] == b'\xE5\x1F':
                fc = bytes([b ^ 0x9C for b in fc[:130]]) + fc[130:]
                fc = fc[::-1]
            return fc
        except zlib.error:
            pass
    return zc

STD_NAME_OP_MAP = {
    "STOP_CODE": 0, "POP_TOP": 1, "ROT_TWO": 2, "ROT_THREE": 3, "DUP_TOP": 4,
    "ROT_FOUR": 5, "NOP": 9, "UNARY_POSITIVE": 10, "UNARY_NEGATIVE": 11,
    "UNARY_NOT": 12, "UNARY_CONVERT": 13, "UNARY_INVERT": 15, "BINARY_POWER": 19,
    "BINARY_MULTIPLY": 20, "BINARY_DIVIDE": 21, "BINARY_MODULO": 22, "BINARY_ADD": 23,
    "BINARY_SUBTRACT": 24, "BINARY_SUBSCR": 25, "BINARY_FLOOR_DIVIDE": 26,
    "BINARY_TRUE_DIVIDE": 27, "INPLACE_FLOOR_DIVIDE": 28, "INPLACE_TRUE_DIVIDE": 29,
    "SLICE+0": 30, "SLICE+1": 31, "SLICE+2": 32, "SLICE+3": 33, "STORE_SLICE+0": 40,
    "STORE_SLICE+1": 41, "STORE_SLICE+2": 42, "STORE_SLICE+3": 43, "DELETE_SLICE+0": 50,
    "DELETE_SLICE+1": 51, "DELETE_SLICE+2": 52, "DELETE_SLICE+3": 53, "STORE_MAP": 54,
    "INPLACE_ADD": 55, "INPLACE_SUBTRACT": 56, "INPLACE_MULTIPLY": 57,
    "INPLACE_DIVIDE": 58, "INPLACE_MODULO": 59, "STORE_SUBSCR": 60, "DELETE_SUBSCR": 61,
    "BINARY_LSHIFT": 62, "BINARY_RSHIFT": 63, "BINARY_AND": 64, "BINARY_XOR": 65,
    "BINARY_OR": 66, "INPLACE_POWER": 67, "GET_ITER": 68, "PRINT_EXPR": 70,
    "PRINT_ITEM": 71, "PRINT_NEWLINE": 72, "PRINT_ITEM_TO": 73, "PRINT_NEWLINE_TO": 74,
    "INPLACE_LSHIFT": 75, "INPLACE_RSHIFT": 76, "INPLACE_AND": 77, "INPLACE_XOR": 78,
    "INPLACE_OR": 79, "BREAK_LOOP": 80, "WITH_CLEANUP": 81, "LOAD_LOCALS": 82,
    "RETURN_VALUE": 83, "IMPORT_STAR": 84, "EXEC_STMT": 85, "YIELD_VALUE": 86,
    "POP_BLOCK": 87, "END_FINALLY": 88, "BUILD_CLASS": 89, "STORE_NAME": 90,
    "DELETE_NAME": 91, "UNPACK_SEQUENCE": 92, "FOR_ITER": 93, "LIST_APPEND": 94,
    "STORE_ATTR": 95, "DELETE_ATTR": 96, "STORE_GLOBAL": 97, "DELETE_GLOBAL": 98,
    "DUP_TOPX": 99, "LOAD_CONST": 100, "LOAD_NAME": 101, "BUILD_TUPLE": 102,
    "BUILD_LIST": 103, "BUILD_SET": 104, "BUILD_MAP": 105, "LOAD_ATTR": 106,
    "COMPARE_OP": 107, "IMPORT_NAME": 108, "IMPORT_FROM": 109, "JUMP_FORWARD": 110,
    "JUMP_IF_FALSE_OR_POP": 111, "JUMP_IF_TRUE_OR_POP": 112, "JUMP_ABSOLUTE": 113,
    "POP_JUMP_IF_FALSE": 114, "POP_JUMP_IF_TRUE": 115, "LOAD_GLOBAL": 116,
    "CONTINUE_LOOP": 119, "SETUP_LOOP": 120, "SETUP_EXCEPT": 121, "SETUP_FINALLY": 122,
    "LOAD_FAST": 124, "STORE_FAST": 125, "DELETE_FAST": 126, "RAISE_VARARGS": 130,
    "CALL_FUNCTION": 131, "MAKE_FUNCTION": 132, "BUILD_SLICE": 133, "MAKE_CLOSURE": 134,
    "LOAD_CLOSURE": 135, "LOAD_DEREF": 136, "STORE_DEREF": 137, "CALL_FUNCTION_VAR": 140,
    "CALL_FUNCTION_KW": 141, "CALL_FUNCTION_VAR_KW": 142, "SETUP_WITH": 143,
    "EXTENDED_ARG": 145, "SET_ADD": 146, "MAP_ADD": 147
}

_OP_MAPS = {
    1: {0x00:74,0x01:15,0x02:28,0x03:56,0x04:40,0x05:41,0x06:42,0x07:43,0x08:11,0x09:24,0x0A:9,0x0C:25,0x0D:66,0x0F:82,0x10:78,0x11:85,0x12:27,0x14:50,0x15:51,0x16:52,0x17:53,0x1A:60,0x1B:71,0x1C:64,0x1D:12,0x1E:73,0x1F:22,0x24:19,0x25:86,0x26:81,0x27:3,0x29:75,0x2A:26,0x2B:54,0x2C:25,0x2D:79,0x2E:63,0x2F:77,0x30:65,0x31:5,0x32:68,0x34:72,0x35:2,0x37:89,0x38:20,0x39:87,0x3B:83,0x3C:80,0x3E:84,0x3F:23,0x40:115,0x41:88,0x43:61,0x45:64,0x46:57,0x47:30,0x48:31,0x49:32,0x4A:33,0x4B:62,0x4E:76,0x4F:60,0x50:4,0x54:10,0x56:1,0x58:55,0x5C:71,0x5D:100,0x5F:122,0x64:93,0x65:143,0x66:125,0x6B:133,0x72:90,0x74:91,0x77:131,0x7A:102,0x7B:134,0x82:107,0x83:146,0x86:137,0x87:106,0x8A:92,0x8B:109,0x8C:95,0x90:96,0x94:108,0x96:116,0x98:126,0x99:120,0x9E:135,0xA0:114,0xAD:136,0xB3:60,0xB7:97,0xBB:146,0xBC:147,0xC0:140,0xC1:141,0xC2:142,0xC5:111,0xC6:132,0xC7:110,0xCA:99,0xCC:145,0xCF:116,0xD2:94,0xD3:115,0xD4:121,0xE2:124,0xE6:112,0xE7:105,0xE8:101,0xE9:115,0xEA:60,0xEB:130,0xF0:104,0xF3:121,0xF7:119,0xFA:113,0xFC:103},
    2: {0x01:83,0x02:81,0x04:62,0x05:74,0x08:30,0x09:31,0x0A:32,0x0B:33,0x0C:9,0x0E:55,0x0F:23,0x10:11,0x12:82,0x14:75,0x15:68,0x17:2,0x1A:24,0x1B:106,0x1C:63,0x1F:66,0x20:27,0x22:4,0x23:85,0x24:87,0x25:57,0x26:54,0x28:77,0x2A:79,0x2B:65,0x2C:20,0x2D:71,0x2E:56,0x2F:60,0x30:71,0x31:27,0x32:12,0x33:3,0x34:71,0x35:22,0x37:89,0x38:86,0x3C:40,0x3D:41,0x3E:42,0x3F:43,0x40:28,0x43:61,0x44:1,0x45:72,0x46:26,0x47:25,0x48:56,0x49:80,0x4A:10,0x4C:24,0x4D:84,0x4E:5,0x4F:88,0x51:19,0x54:50,0x55:51,0x56:52,0x57:53,0x58:64,0x59:76,0x5B:73,0x5C:78,0x5E:137,0x61:105,0x64:110,0x66:99,0x6E:111,0x72:145,0x74:131,0x78:121,0x82:95,0x8B:133,0x8D:140,0x8E:141,0x8F:142,0x94:110,0x95:90,0x9E:92,0xAC:134,0xAF:102,0xB0:114,0xB9:122,0xBC:130,0xBE:116,0xC0:125,0xC2:132,0xC4:112,0xC6:146,0xCA:96,0xCC:101,0xCD:136,0xD1:104,0xD2:106,0xD4:97,0xD5:107,0xD6:119,0xD9:135,0xDB:109,0xDC:143,0xDD:91,0xDF:100,0xE5:103,0xE8:113,0xE9:120,0xEB:108,0xEE:93,0xF2:115,0xF3:111,0xF4:124,0xF7:94,0xFD:116},
    3: {0x00:61,0x01:9,0x02:19,0x03:82,0x04:20,0x06:65,0x08:84,0x09:11,0x0A:26,0x0B:25,0x0C:50,0x0D:51,0x0E:52,0x0F:53,0x10:66,0x11:28,0x14:24,0x16:62,0x18:74,0x19:86,0x1A:78,0x1C:40,0x1D:41,0x1E:42,0x1F:43,0x29:75,0x2A:71,0x2B:79,0x2D:2,0x2E:3,0x2F:88,0x30:27,0x31:54,0x32:1,0x34:22,0x35:72,0x36:63,0x37:12,0x3B:68,0x3C:83,0x3F:80,0x43:10,0x44:76,0x45:77,0x47:81,0x48:64,0x49:57,0x4A:87,0x4C:73,0x4E:23,0x50:89,0x51:56,0x52:55,0x53:4,0x55:30,0x56:31,0x57:32,0x58:33,0x5A:60,0x60:120,0x65:107,0x67:112,0x69:134,0x6B:116,0x6C:122,0x6D:103,0x6E:94,0x72:93,0x76:119,0x77:110,0x78:95,0x7E:131,0x88:111,0x94:137,0x97:140,0x98:141,0x99:142,0x9B:146,0x9E:97,0xA9:91,0xAC:105,0xAD:143,0xB2:92,0xB7:96,0xB9:108,0xBE:102,0xC0:124,0xC7:104,0xC8:115,0xD3:136,0xD4:101,0xD6:100,0xD7:132,0xDB:130,0xDE:125,0xDF:109,0xE0:99,0xE4:90,0xE7:106,0xF1:114,0xF4:116,0xF6:135,0xF7:121,0xFE:113},
    4: {0x00:9,0x01:5,0x02:76,0x03:15,0x04:40,0x05:41,0x06:42,0x07:43,0x08:25,0x09:72,0x0A:21,0x0B:19,0x0D:75,0x0E:27,0x0F:68,0x10:56,0x11:87,0x12:86,0x13:28,0x14:54,0x15:20,0x17:71,0x18:57,0x1A:81,0x1B:30,0x1C:31,0x1D:32,0x1E:33,0x1F:63,0x23:60,0x27:66,0x28:88,0x29:77,0x2A:21,0x2B:4,0x2C:10,0x2E:80,0x2F:73,0x34:3,0x35:82,0x36:24,0x3B:62,0x3C:78,0x3E:50,0x3F:51,0x40:52,0x41:53,0x43:89,0x44:79,0x46:11,0x47:24,0x48:65,0x49:23,0x4B:22,0x4D:55,0x4E:66,0x4F:85,0x50:2,0x51:83,0x52:84,0x53:1,0x56:12,0x57:26,0x58:74,0x59:61,0x5C:64,0x60:137,0x6B:93,0x6E:99,0x6F:104,0x74:122,0x79:110,0x7B:113,0x84:101,0x85:136,0x86:97,0x87:91,0x88:116,0x89:106,0x8C:124,0x90:131,0x9F:125,0xA1:105,0xA2:126,0xA3:109,0xA6:134,0xA7:119,0xAF:103,0xB6:130,0xBD:146,0xC2:143,0xC3:111,0xC9:90,0xD0:120,0xD1:121,0xD4:95,0xD5:112,0xD6:135,0xD8:94,0xDA:114,0xDE:100,0xDF:147,0xE1:96,0xE4:107,0xE5:102,0xE9:140,0xEA:141,0xEB:142,0xF0:116,0xF3:115,0xF7:132,0xFA:108,0xFC:133,0xFE:92}
}

_NAME_OP_CACHE = {}


def _get_name_op_map(version):
    if version < 0:
        version = 1
    if version not in _NAME_OP_CACHE:
        m2s = _OP_MAPS.get(version, _OP_MAPS[1])
        m = STD_NAME_OP_MAP.copy()
        for name, std in STD_NAME_OP_MAP.items():
            for mo, so in m2s.items():
                if so == std:
                    m[name] = mo
                    break
        _NAME_OP_CACHE[version] = m
    return _NAME_OP_CACHE[version]


def _get_std_op_map(version):
    return _OP_MAPS.get(version if version >= 0 else 1, _OP_MAPS[1])

_U16 = struct.Struct('<H')
_U32 = struct.Struct('<i')
_U64 = struct.Struct('<q')
_DBL = struct.Struct('<d')

_CO_CACHE = {}

def _get_co_sets(ops):
    key = id(ops)
    if key not in _CO_CACHE:
        name_ops = frozenset(
            ops[n] for n in (
                'STORE_NAME', 'LOAD_NAME', 'DELETE_NAME', 'STORE_GLOBAL',
                'LOAD_GLOBAL', 'DELETE_GLOBAL', 'STORE_ATTR', 'LOAD_ATTR', 'DELETE_ATTR'
            ) if n in ops
        )
        fast_ops = frozenset(
            ops[n] for n in ('STORE_FAST', 'LOAD_FAST', 'DELETE_FAST') if n in ops
        )
        invalid = frozenset(
            op for nm, op in ops.items()
            if nm.startswith(('INPLACE', 'BINARY', 'ROT'))
        )
        _CO_CACHE[key] = (name_ops, fast_ops, invalid)
    return _CO_CACHE[key]

_NULL = object()

class _McsRC4:
    __slots__ = ('s', 'i', 'j')

    def __init__(self, key):
        s = list(range(256))
        kl = len(key)
        j = 0
        for i in range(256):
            j = (j + s[i] + key[i % kl]) & 0xFF
            s[i], s[j] = s[j], s[i]
        self.s = s
        self.i = self.j = 0

    def decrypt(self, data):
        s = self.s
        i = self.i
        j = self.j
        d = bytearray(data)
        for k in range(len(d)):
            i = (i + 1) & 0xFF
            j = (j + s[i]) & 0xFF
            si, sj = s[i], s[j]
            s[i], s[j] = sj, si
            d[k] ^= s[(si + sj) & 0xFF]
        self.i = i
        self.j = j
        return bytes(d)


class McsMarshal:
    K2 = b"\xa7\x0d\x37\x7a"
    K3 = b"\x8d\x06\xe8\xc8\xb7\xd7\xb7\x28\x46\x51\xae\x04"
    __slots__ = ('d', 'p', 'r', 'rg', 'mv', 'n')

    def __init__(self, data, remove_garbage=True):
        if isinstance(data, bytearray):
            data = bytes(data)
        self.d = data
        self.mv = memoryview(data)
        self.n = len(data)
        self.p = 0
        self.r = []
        self.rg = remove_garbage

    def _b(self):
        v = self.mv[self.p]
        self.p += 1
        return v

    def _h(self):
        v = _U16.unpack_from(self.d, self.p)[0]
        self.p += 2
        return v

    def _i(self):
        v = _U32.unpack_from(self.d, self.p)[0]
        self.p += 4
        return v

    def _l(self):
        sz = self._i()
        if sz == 0:
            return 0
        n = abs(sz)
        res = 0
        for i in range(n):
            res |= (self._h() & 0x7FFF) << (i * 15)
        return res if sz > 0 else -res

    def _s(self, sz=None):
        if sz is None:
            sz = self._i()
        if sz < 0:
            return b""
        end = self.p + sz
        if end > self.n:
            sz = self.n - self.p
            end = self.n
        v = self.mv[self.p:end].tobytes()
        self.p = end
        return v

    def _rc4(self, key):
        return _McsRC4(key).decrypt(self._s())

    def _obj(self):
        t = self._b()
        if t == 48:
            return _NULL
        if t in (78, 110):
            return None
        if t == 84:
            return True
        if t == 70:
            return False
        if t == 46:
            return Ellipsis
        if t == 83:
            class _StopIter:
                pass
            return _StopIter()
        if t == 105:
            return self._i()
        if t == 73:
            v = _U64.unpack_from(self.d, self.p)[0]
            self.p += 8
            return v
        if t in (108, 76):
            return self._l()
        if t == 102:
            return float(self._s(self._b()))
        if t == 103:
            v = _DBL.unpack_from(self.d, self.p)[0]
            self.p += 8
            return v
        if t == 115:
            return self._s()
        if t == 116:
            s = self._s()
            self.r.append(s)
            return s
        if t == 117:
            return self._s().decode('utf-8', 'ignore')
        if t == 82:
            idx = self._i()
            return self.r[idx] if idx < len(self.r) else None
        if t == 40:
            return tuple(self._obj() for _ in range(self._i()))
        if t == 91:
            return [self._obj() for _ in range(self._i())]
        if t in (60, 62):
            items = [self._obj() for _ in range(self._i())]
            return frozenset(items) if t == 62 else set(items)
        if t == 123:
            d = {}
            while True:
                k = self._obj()
                if k is _NULL:
                    break
                d[k] = self._obj()
            return d
        if t in (109, 49, 23, 26, 29):
            key = self.K2 if t in (23, 26, 29) else self.K3
            dec = self._rc4(key)
            if t == 26:
                self.r.append(dec)
            if t == 29:
                return dec.decode('utf-8', 'ignore')
            return dec
        if t == 98:
            dec = self._rc4(self.K3)
            self.r.append(dec)
            return dec
        if t in (8, 14, 15):
            raw = bytearray(self._s())
            for i in range(len(raw)):
                raw[i] ^= 0x8D
            res = bytes(raw)
            if t == 15:
                self.r.append(res)
            return res
        if t in (99, 77, 111, 97):
            return self._co(t)
        raise ValueError("Unknown tag %d at %d" % (t, self.p - 1))

    def _co(self, tag):
        if tag == 99:
            obj = {'argcount': self._i(), 'nlocals': self._i(), 'stacksize': self._i(),
                   'flags': self._i(), 'code': self._obj(), 'consts': self._obj(),
                   'names': self._obj(), 'varnames': self._obj(), 'freevars': self._obj(),
                   'cellvars': self._obj(), 'filename': self._obj(), 'name': self._obj(),
                   'firstlineno': self._i(), 'lnotab': self._obj(), 'magic': None, 'version': 1}
        elif tag == 77:
            obj = {'argcount': self._i(), 'lnotab': self._obj(), 'cellvars': self._obj(),
                   'firstlineno': self._i(), 'varnames': self._obj(), 'consts': self._obj(),
                   'name': self._obj(), 'stacksize': self._i(), 'freevars': self._obj(),
                   'names': self._obj(), 'code': self._obj(), 'flags': self._i(),
                   'filename': self._obj(), 'nlocals': self._i(), 'magic': self._i(), 'version': 4}
        elif tag == 111:
            obj = {'nlocals': self._i(), 'flags': self._i(), 'consts': self._obj(),
                   'stacksize': self._i(), 'varnames': self._obj(), 'argcount': self._i(),
                   'cellvars': self._obj(), 'names': self._obj(), 'freevars': self._obj(),
                   'name': self._obj(), 'code': self._obj(), 'firstlineno': self._i(),
                   'lnotab': self._obj(), 'magic': self._i(), 'filename': self._obj(), 'version': 2}
        elif tag == 97:
            obj = {'lnotab': self._obj(), 'varnames': self._obj(), 'flags': self._i(),
                   'freevars': self._obj(), 'cellvars': self._obj(), 'filename': self._obj(),
                   'stacksize': self._i(), 'firstlineno': self._i(), 'consts': self._obj(),
                   'argcount': self._i(), 'code': self._obj(), 'nlocals': self._i(),
                   'name': self._obj(), 'names': self._obj(), 'magic': self._i(), 'version': 3}
        else:
            obj = {}

        if self.rg and isinstance(obj, dict):
            v = obj.get('version', 1)
            ops = _get_name_op_map(v)
            code = obj.get('code', b'')
            names = obj.get('names', [])
            consts = obj.get('consts', [])
            varnames = obj.get('varnames', [])
            score = 0
            if not code:
                score = 0
            else:
                nop = ops.get('NOP', 9)
                name_ops, fast_ops, invalid = _get_co_sets(ops)
                cst = ops.get('LOAD_CONST')
                n_consts = len(consts)
                n_names = len(names)
                n_varnames = len(varnames)
                n_code = len(code)
                cv = memoryview(code)
                i = 0
                first = True
                while i < n_code:
                    op = cv[i]
                    if op == nop:
                        i += 1
                        continue
                    if first and op in invalid:
                        score = 9999
                        break
                    first = False
                    if op >= 93:
                        if i + 2 >= n_code:
                            score += 100
                            break
                        arg = cv[i + 1] | (cv[i + 2] << 8)
                        if op == cst and arg >= n_consts:
                            score += 60
                        elif op in name_ops and arg >= n_names:
                            score += 60
                        elif op in fast_ops and arg >= n_varnames:
                            score += 60
                        i += 3
                    else:
                        i += 1
                if n_code > 10 and (code.count(nop) / n_code) > 0.3:
                    score += 40
            if score > 100:
                obj['code'] = bytes([ops.get('LOAD_CONST',100),0,0,ops.get('RETURN_VALUE',83)])
                obj['consts'] = (None,)
                obj['names'] = ()
                obj['varnames'] = ()
                obj['argcount'] = 0
                obj['nlocals'] = 0
                obj['stacksize'] = 1
                obj['lnotab'] = b''
        return obj

class _FakeFile:
    __slots__ = ('buf',)
    def __init__(self):
        self.buf = bytearray()
    def write(self, b):
        if isinstance(b, str):
            b = b.encode('latin-1')
        self.buf.extend(b)
    def get(self):
        return bytes(self.buf)


_I32 = struct.Struct('<i')
_U8 = struct.Struct('B')
_TYPE_TAG = {tuple: b'(', list: b'[', frozenset: b'>', set: b'<'}

def _w_long(v, f):
    w = f.write
    w(b'l')
    if v == 0:
        w(_I32.pack(0))
        return
    s = 1 if v >= 0 else -1
    a = abs(v)
    digits = []
    while a:
        digits.append(a & 0x7FFF)
        a >>= 15
    w(_I32.pack(s * len(digits)))
    for d in digits:
        w(_U16.pack(d))


def _w_obj(obj, f):
    w = f.write
    if obj is None:
        w(b'N')
    elif obj is True:
        w(b'T')
    elif obj is False:
        w(b'F')
    elif obj is Ellipsis:
        w(b'.')
    elif isinstance(obj, int):
        if -2147483648 <= obj <= 2147483647:
            w(b'i')
            w(_I32.pack(obj))
        else:
            _w_long(obj, f)
    elif isinstance(obj, float):
        s = repr(obj).encode()
        w(b'f')
        w(_U8.pack(len(s)))
        w(s)
    elif isinstance(obj, (bytes, bytearray)):
        w(b's')
        w(_I32.pack(len(obj)))
        w(bytes(obj) if isinstance(obj, bytearray) else obj)
    elif isinstance(obj, str):
        b = obj.encode('utf-8')
        w(b's')
        w(_I32.pack(len(b)))
        w(b)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        w(_TYPE_TAG.get(type(obj), b'<'))
        w(_I32.pack(len(obj)))
        for it in obj:
            _w_obj(it, f)
    elif isinstance(obj, dict) and 'magic' in obj:
        w(b'c')
        w(_I32.pack(obj['argcount']))
        w(_I32.pack(obj['nlocals']))
        w(_I32.pack(obj['stacksize']))
        w(_I32.pack(obj['flags']))
        mc = bytearray(obj['code'])
        vm = obj.get('version', 1)
        omap = _get_std_op_map(vm)
        nc = bytearray()
        i = 0
        n = len(mc)
        while i < n:
            mo = mc[i]
            if mo >= 93:
                if i + 2 < n:
                    ar = mc[i+1] | (mc[i+2] << 8)
                    st = 3
                else:
                    ar = 0
                    st = n - i
            else:
                ar = None
                st = 1
            so = omap.get(mo, mo)
            nc.append(so)
            if so >= 90:
                a = ar if ar is not None else 0
                nc.extend([a & 0xFF, (a >> 8) & 0xFF])
            i += st
        _w_obj(bytes(nc), f)
        _w_obj(tuple(obj['consts']), f)
        _w_obj(tuple(obj['names']), f)
        _w_obj(tuple(obj['varnames']), f)
        _w_obj(tuple(obj['freevars']), f)
        _w_obj(tuple(obj['cellvars']), f)
        _w_obj(obj['filename'], f)
        _w_obj(obj['name'], f)
        w(_I32.pack(obj['firstlineno']))
        _w_obj(obj['lnotab'], f)
    elif isinstance(obj, dict):
        w(b'{')
        for k, v in obj.items():
            _w_obj(k, f)
            _w_obj(v, f)
        w(b'0')
    else:
        w(b'N')


def _restore(data):
    d = _decrypt_data(data)
    p = McsMarshal(d)
    r = p._obj()
    f = _FakeFile()
    f.write(b"\x03\xf3\x0d\x0a\x00\x00\x00\x00")
    _w_obj(r, f)
    return f.get()


# =============================================================================
# 6. MCPK hash + unpack
# =============================================================================
M1, M2 = 0x267B0B11, 0xBDEB77DE
M3, M4, M5 = 0x02040801, 0x7D7EBBDE, 0x00804021
HI, H2I, RI = 933775118, 2002301995, 0xF4FA8928


def _uh(h1, h2, rot, chunk):
    x1, x2 = (h1 ^ chunk) & 0xFFFFFFFF, (h2 ^ chunk) & 0xFFFFFFFF
    k1 = (((rot ^ M1) + x2) & 0xFFFFFFFF) & M2 | M3
    p1 = x1 * k1
    s1 = ((p1 >> 32) & 0xFFFFFFFF) + (1 if (p1 >> 32) & 0xFFFFFFFF != 0 else 0) + (p1 & 0xFFFFFFFF)
    nh1 = (s1 + (s1 >> 32)) & 0xFFFFFFFF
    k2 = (((rot ^ M1) + x1) & 0xFFFFFFFF) & M4 | M5
    p2 = x2 * k2
    s2 = (p2 & 0xFFFFFFFF) + 2 * ((p2 >> 32) & 0xFFFFFFFF)
    nh2 = (s2 + 2 * (s2 >> 32)) & 0xFFFFFFFF
    return nh1, nh2


def _fh(h1, h2, rot):
    f2, f1 = (h2 ^ 0x9BE74448) & 0xFFFFFFFF, (h1 ^ 0x9BE74448) & 0xFFFFFFFF
    rf1 = ((rot << 1) & 0xFFFFFFFF) | (rot >> 31)
    k1 = (rf1 ^ M1) & 0xFFFFFFFF
    t1 = ((k1 + f2) & 0xFFFFFFFF) & M2 | M3
    p1 = f1 * t1
    s1 = ((p1 >> 32) & 0xFFFFFFFF) + (1 if (p1 >> 32) & 0xFFFFFFFF != 0 else 0) + (p1 & 0xFFFFFFFF)
    y1 = ((s1 & 0xFFFFFFFF) + (s1 >> 32)) ^ 0x66F42C48
    t2 = ((k1 + f1) & 0xFFFFFFFF) & M4 | M5
    p2 = f2 * t2
    s2 = (p2 & 0xFFFFFFFF) + 2 * ((p2 >> 32) & 0xFFFFFFFF)
    y2 = ((s2 + 2 * (s2 >> 32)) & 0xFFFFFFFF) ^ 0x66F42C48
    rf2 = ((rot << 2) & 0xFFFFFFFF) | (rot >> 30)
    k2 = (rf2 ^ M1) & 0xFFFFFFFF
    t3 = ((k2 + y2) & 0xFFFFFFFF) & M2 | M3
    p3 = y1 * t3
    s3 = ((p3 >> 32) & 0xFFFFFFFF) + (1 if (p3 >> 32) & 0xFFFFFFFF != 0 else 0) + (p3 & 0xFFFFFFFF)
    p1 = ((s3 & 0xFFFFFFFF) + (s3 >> 32)) & 0xFFFFFFFF
    t4 = ((k2 + y1) & 0xFFFFFFFF) & M4 | M5
    p4 = y2 * t4
    p4_64 = p4 & 0xFFFFFFFFFFFFFFFF
    s4 = (p4_64 & 0xFFFFFFFF) + 2 * ((p4_64 >> 32) & 0xFFFFFFFF) + (p4_64 >> 63)
    p2 = ((s4 & 0xFFFFFFFF) + 2 * (s4 >> 32)) & 0xFFFFFFFF
    return (p1 ^ p2) & 0xFFFFFFFF


def _hd(data):
    if isinstance(data, str):
        data = data.encode('ascii')
    ls = data.rfind(b'/')
    if ls != -1:
        data = data[:ls]
    else:
        return 0
    if not data:
        return 0
    h1, h2, rot = HI, H2I, RI
    i = 0
    L = len(data)
    while i + 4 <= L:
        rot = ((rot << 1) & 0xFFFFFFFF) | (rot >> 31)
        h1, h2 = _uh(h1, h2, rot, struct.unpack('<I', data[i:i + 4])[0])
        i += 4
    if i < L:
        rot = ((rot << 1) & 0xFFFFFFFF) | (rot >> 31)
        chunk = 0
        for j in range(L - i):
            chunk |= data[i + j] << (j * 8)
        h1, h2 = _uh(h1, h2, rot, chunk)
    return _fh(h1, h2, rot)


def _hf(data):
    if isinstance(data, str):
        data = data.encode('ascii')
    h1, h2, rot = HI, H2I, RI
    L = len(data)
    idx = 0
    if idx >= L or data[idx] == 0:
        return _fh(h1, h2, rot)
    while idx < L:
        rot = ((rot << 1) & 0xFFFFFFFF) | (rot >> 31)
        chunk = 0
        for j in range(4):
            if idx < L and data[idx] != 0:
                chunk |= data[idx] << (j * 8)
                idx += 1
            else:
                h1, h2 = _uh(h1, h2, rot, chunk)
                return _fh(h1, h2, rot)
        h1, h2 = _uh(h1, h2, rot, chunk)
    return _fh(h1, h2, rot)


def _decompress_maybe(raw):
    head = raw[:2]
    if head == b'\x78\x9C' or head == b'\x78\xDA':
        try:
            return zlib.decompress(raw)
        except zlib.error:
            pass
    return raw.tobytes() if isinstance(raw, memoryview) else raw


def unpack_to_memory(file_path):
    """Returns list of (rel_path_or_None, raw_bytes) and has_contents flag."""
    with open(file_path, 'rb') as f:
        data = f.read()
    if data[:4] != b'MCPK':
        raise ValueError("Not a MCPK file")
    mv = memoryview(data)
    dto = struct.unpack_from('<I', mv, 12)[0]
    ibo = struct.unpack_from('<I', mv, 16)[0]

    dir_entries = []
    pos = dto
    while pos < ibo:
        dir_entries.append(struct.unpack_from('<III', mv, pos))
        pos += 12

    max_rel = max((off for _, off, _ in dir_entries), default=0)
    last_cnt = 0
    for _, off, cnt in dir_entries:
        if off == max_rel:
            last_cnt = cnt
            break
    dbo = ibo + max_rel + last_cnt * 16

    dir_map = {}
    for dh, off, cnt in dir_entries:
        files = {}
        pos = ibo + off
        for _ in range(cnt):
            fh, fo, cs, us = struct.unpack_from('<IIII', mv, pos)
            files[fh] = (fo, cs, us)
            pos += 16
        dir_map[dh] = files

    cjh = _hf("contents.json")
    rmh = _hf("redirect.mcs")
    results = []
    has_contents = False

    if 0 in dir_map and cjh in dir_map[0]:
        fo, cs, _ = dir_map[0][cjh]
        cdata = _decompress_maybe(mv[dbo + fo:dbo + fo + cs])
        if isinstance(cdata, memoryview):
            cdata = cdata.tobytes()
        try:
            flist = json.loads(cdata.decode('utf-8'))
            if isinstance(flist, dict):
                flist = flist.get("content", flist)
            for item in flist:
                ps = item.get("path", "").replace('\\', '/')
                if not ps:
                    continue
                dh = _hd(ps)
                fn = ps.rsplit('/', 1)[1] if '/' in ps else ps
                fh = _hf(fn)
                if dh in dir_map and fh in dir_map[dh]:
                    fo, cs, _ = dir_map[dh][fh]
                    raw = _decompress_maybe(mv[dbo + fo:dbo + fo + cs])
                    if isinstance(raw, memoryview):
                        raw = raw.tobytes()
                    results.append((ps, raw))
            has_contents = bool(results)
        except Exception:
            pass

    if not has_contents:
        for dh, files in dir_map.items():
            for fh, (fo, cs, us) in files.items():
                raw = _decompress_maybe(mv[dbo + fo:dbo + fo + cs])
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                results.append((None, raw))

    return results, has_contents

def _process_one(args):
    """Process a single file. Returns (success: bool, out_path_or_name: str, error_msg: str)."""
    idx, rel_path, raw_bytes, output_dir, has_contents, pycdc_path = args
    temp_pyc = None
    out_path = rel_path or f"unknown_{idx}.mcs"
    pyc = None
    try:
        temp_pyc = os.path.join(_TMPDIR, f"mcp2py_{os.getpid()}_{idx}.pyc")

        decrypted = _decrypt_data(raw_bytes)
        parser = McsMarshal(decrypted)
        root = parser._obj()

        if has_contents and rel_path:
            out_path = os.path.join(output_dir, rel_path.replace('.mcs', '.py'))
        elif isinstance(root, dict) and root.get('filename'):
            fname = root['filename']
            if isinstance(fname, bytes):
                fname = fname.decode('utf-8', 'ignore')
            out_path = os.path.join(output_dir, fname.replace('.mcs', '.py'))
            if not out_path.endswith('.py'):
                out_path += '.py'
        else:
            out_path = os.path.join(output_dir, f"unknown_{idx}.py")

        f = _FakeFile()
        f.write(b"\x03\xf3\x0d\x0a\x00\x00\x00\x00")
        _w_obj(root, f)
        pyc = f.get()

        with open(temp_pyc, 'wb') as tf:
            tf.write(pyc)

        proc = subprocess.run(
            [pycdc_path, temp_pyc],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if proc.returncode != 0:
            raise RuntimeError(f"pycdc exit {proc.returncode}: {proc.stderr.strip()}")
        py_src = proc.stdout

        d = os.path.dirname(out_path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as wf:
            wf.write(py_src)
        return True, out_path, ""
    except Exception as e:
        if pyc is not None:
            pyc_path = out_path.replace('.py', '.pyc')
            if not pyc_path.endswith('.pyc'):
                pyc_path += '.pyc'
            try:
                d = os.path.dirname(pyc_path)
                if d and not os.path.isdir(d):
                    os.makedirs(d, exist_ok=True)
                with open(pyc_path, 'wb') as f:
                    f.write(pyc)
            except Exception:
                pass
        return False, out_path, str(e)
    finally:
        if temp_pyc and os.path.exists(temp_pyc):
            try:
                os.remove(temp_pyc)
            except OSError:
                pass

def _find_pycdc():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'pycdc', 'Release', 'pycdc.exe'),
        os.path.join(script_dir, 'pycdc', 'pycdc.exe'),
        os.path.join(script_dir, 'pycdc', 'pycdc'),
        os.path.join(script_dir, 'pycdc.exe'),
        os.path.join(script_dir, 'pycdc'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    for path_env in os.environ.get('PATH', '').split(os.pathsep):
        for name in ('pycdc.exe', 'pycdc'):
            p = os.path.join(path_env.strip('"'), name)
            if os.path.isfile(p):
                return p
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python mcpdec.py <input.mcp> [output_dir]")
        sys.exit(1)

    mcp_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(mcp_path))[0] + "_py"

    if not os.path.isfile(mcp_path):
        print(f"[!] File not found: {mcp_path}")
        sys.exit(1)

    pycdc_path = _find_pycdc()
    if pycdc_path:
        print(f"[+] Found pycdc: {pycdc_path}")
    else:
        print("[!] pycdc not found. Searched: script_dir/pycdc/, PATH")
        print("[!] Will output .pyc files instead.")

    t0 = time.time()

    print(f"[+] Unpacking {mcp_path} ...")
    files, has_contents = unpack_to_memory(mcp_path)
    print(f"[+] Unpacked {len(files)} files in {time.time()-t0:.1f}s")

    if not pycdc_path:
        ok = 0
        for idx, (rp, raw) in enumerate(files):
            try:
                pyc = _restore(raw)
                if has_contents and rp:
                    out = os.path.join(output_dir, rp.replace('.mcs', '.pyc'))
                else:
                    out = os.path.join(output_dir, f"file_{idx}.pyc")
                d = os.path.dirname(out)
                if d and not os.path.isdir(d):
                    os.makedirs(d, exist_ok=True)
                with open(out, 'wb') as f:
                    f.write(pyc)
                ok += 1
            except Exception:
                pass
        print(f"[+] Output {ok} .pyc files to {output_dir}")
        return

    cpu = multiprocessing.cpu_count()
    workers = min(cpu // 2 + 1, 8)

    # Sort by raw_bytes length descending for better load balancing (avoid stragglers)
    files = sorted(files, key=lambda x: len(x[1]), reverse=True)

    args_list = [(i, rp, raw, output_dir, has_contents, pycdc_path) for i, (rp, raw) in enumerate(files)]

    # Pre-create all output directories to avoid per-file makedirs in workers
    dirs_needed = set()
    for i, (rp, raw) in enumerate(files):
        if has_contents and rp:
            out_path = os.path.join(output_dir, rp.replace('.mcs', '.py'))
        else:
            out_path = os.path.join(output_dir, f"unknown_{i}.py")
        d = os.path.dirname(out_path)
        if d:
            dirs_needed.add(d)
    for d in dirs_needed:
        os.makedirs(d, exist_ok=True)

    total_ok = total_fail = 0
    failed = []
    print(f"[*] Starting {workers} workers for {len(files)} files ...")
    t1 = time.time()
    if workers > 1 and len(files) > 1:
        chunksize = max(1, min(20, len(args_list) // workers // 2))
        with multiprocessing.Pool(processes=workers) as pool:
            for ok, out_path, err in pool.imap_unordered(_process_one, args_list, chunksize=chunksize):
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1
                    failed.append((out_path, err))
    else:
        for args in args_list:
            ok, out_path, err = _process_one(args)
            if ok:
                total_ok += 1
            else:
                total_fail += 1
                failed.append((out_path, err))

    _cleanup_orphan_temps()
    elapsed = time.time() - t0
    proc_time = time.time() - t1
    print(f"\n{'='*60}")
    print(f"[+] DONE: {output_dir}")
    print(f"    Files: {total_ok} OK, {total_fail} fail / {len(files)} total")
    print(f"    Total time:  {elapsed:.1f}s")
    print(f"    Process time: {proc_time:.1f}s")
    if failed:
        print(f"\n[!] Failed files ({len(failed)}):")
        for out_path, err in failed:
            print(f"    - {os.path.basename(out_path)}: {err}")
    print(f"{'='*60}")


def _cleanup_orphan_temps():
    tmp = tempfile.gettempdir()
    for fn in os.listdir(tmp):
        if fn.startswith('mcp2py_') and fn.endswith('.pyc'):
            try:
                os.remove(os.path.join(tmp, fn))
            except OSError:
                pass


if __name__ == '__main__':
    main()
