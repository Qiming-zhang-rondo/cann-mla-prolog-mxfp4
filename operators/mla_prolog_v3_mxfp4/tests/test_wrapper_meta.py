"""Execute the real Python metadata helpers without loading the NPU extension."""
import ast
from pathlib import Path
from typing import List, Optional
import unittest
import torch


class WrapperMetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[3]/'attention/mla_prolog_v3/torch_extension/mla_prolog.py'
        tree=ast.parse(path.read_text())
        names={'_has_defined','_resolve_do_rope','_resolve_rope_dim','_is_full_quant_kv','_is_hifloat8','_query_shape','_meta_outputs'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        namespace={'torch':torch,'Optional':Optional,'List':List,'DIM_2':2,'DIM_3':3,'MODE_1':1,
                   'MODE_2':2,'MODE_3':3,'MODE_4':4,'MODE_5':5,'MODE_MXFP4':6,'FP8_E4M3_BLOCK_SIZE':32}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        cls.meta=staticmethod(namespace['_meta_outputs'])

    def test_mode6_logical_norm_from_packed_nz(self):
        for flag in (False,True):
            p=torch.empty(17,3072,dtype=torch.uint8,device='meta')
            dq=torch.empty(32,384,16,32,dtype=torch.uint8,device='meta')
            uq=torch.empty(16,128,16,32,dtype=torch.uint8,device='meta')
            uk=torch.empty(4,192,512,dtype=torch.bfloat16,device='meta')
            dkv=torch.empty(9,384,16,32,dtype=torch.uint8,device='meta')
            rope=torch.empty(17,64,dtype=torch.bfloat16,device='meta')
            out=self.meta(p,dq,uq,uk,dkv,rope,rope,torch.empty(0,device='meta'),flag,6,3,None,None)
            self.assertEqual(tuple(out[0].shape),(17,4,512))
            self.assertEqual(tuple(out[1].shape),(17,4,64))
            self.assertEqual(tuple(out[3].shape),(17,2048) if flag else (0,))
            self.assertEqual(out[3].dtype,torch.bfloat16)
            self.assertEqual(out[4].numel(),0)
            self.assertEqual(out[4].dtype,torch.float32)


if __name__=='__main__':unittest.main()
