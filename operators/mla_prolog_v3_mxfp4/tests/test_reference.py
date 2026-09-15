import unittest
import numpy as np
from reference import (pack_codes, unpack_codes, decode_e2m1, pack_nz_from_codes,
                       unpack_nz_codes, prolog_to_matmul_scale, matmul_to_prolog_scale,
                       dequant_rows, prolog_rope, rmsnorm_bf16)


class ContractTests(unittest.TestCase):
    def test_all_fp4_codes_and_negative_zero(self):
        codes = np.arange(16, dtype=np.uint8).reshape(1,16)
        np.testing.assert_array_equal(unpack_codes(pack_codes(codes)), codes)
        x = decode_e2m1(pack_codes(codes))[0]
        self.assertTrue(np.signbit(x[8])); self.assertFalse(np.signbit(x[0]))
        self.assertEqual(x[15], -6)

    def test_nz_cross_k_n_tile(self):
        c = (np.arange(128*192).reshape(128,192)*7 % 16).astype(np.uint8)
        nz = pack_nz_from_codes(c)
        self.assertEqual(nz.shape,(3,8,16,32))
        np.testing.assert_array_equal(unpack_nz_codes(nz), c)
        for k,n in ((0,0),(15,63),(16,64),(127,191)):
            byte=int(nz[n//64,k//16,k%16,(n%64)//2])
            self.assertEqual((byte >> (4*(n%2))) & 15, int(c[k,n]))

    def test_nonuniform_scale_pair_order(self):
        s = np.arange(8*6,dtype=np.uint8).reshape(8,6)+110
        q = prolog_to_matmul_scale(s)
        np.testing.assert_array_equal(matmul_to_prolog_scale(q),s)
        for n in range(8):
            for g in range(6):self.assertEqual(q[g//2,n,g%2],s[n,g])
        self.assertFalse(np.array_equal(q.reshape(8,6),s))

    def test_dequant_uses_each_k32_scale(self):
        p=pack_codes(np.full((2,64),2,dtype=np.uint8))
        d=dequant_rows(p,np.array([[127,128],[129,126]],dtype=np.uint8))
        np.testing.assert_array_equal(d[:,[0,31,32,63]],[[1,1,2,2],[4,4,.5,.5]])

    def test_rope_matches_glm_interleave_after_layout_permutation(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(3,64))
        a=rng.uniform(-2,2,size=(3,32));c=np.cos(a);s=np.sin(a)
        y=prolog_rope(x,np.concatenate((c,c),-1),np.concatenate((s,s),-1))
        expected=np.empty_like(x)
        expected[:,::2]=x[:,::2]*c-x[:,1::2]*s
        expected[:,1::2]=x[:,1::2]*c+x[:,::2]*s
        np.testing.assert_allclose(y,np.concatenate((expected[:,::2],expected[:,1::2]),-1))
        np.testing.assert_allclose(np.sum(y*y,-1),np.sum(x*x,-1))

    def test_norm_zero_and_sign(self):
        np.testing.assert_array_equal(rmsnorm_bf16(np.zeros((2,64)),np.ones(64)),0)
        x=np.arange(64,dtype=np.float32)[None,:]/32
        np.testing.assert_array_equal(rmsnorm_bf16(-x,np.ones(64)),-rmsnorm_bf16(x,np.ones(64)))

    def test_bad_layout_rejected(self):
        with self.assertRaises(ValueError):pack_nz_from_codes(np.zeros((32,64),np.uint8))
        with self.assertRaises(ValueError):prolog_to_matmul_scale(np.zeros((8,3),np.uint8))


if __name__=='__main__':unittest.main()
