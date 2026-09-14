#!/usr/bin/env python3
"""Real A5 mode-6 V3 vs native MXFP4 composition, no model weights required."""
import argparse
import importlib
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import torch
import torch_npu
cann_ops_transformer = importlib.import_module(
    os.environ.get('MLA_MXFP4_TORCH_PACKAGE', 'cann_ops_transformer_mla_mxfp4'))

from reference import pack_nz_from_codes, unpack_codes, prolog_to_matmul_scale, dequant_rows

ACCURACY_RECORDS = []
CASE_CONTEXT = {}


def quant_rows(x):
    p,s=torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2, round_mode='round')
    return p.view(torch.uint8),s.reshape(x.shape[0],-1).view(torch.float8_e8m0fnu)


def weight(k,n,generator,device):
    # Channel AND K-block variation exposes wrong transpose/view scale producers.
    w=torch.randn(n,k,generator=generator,dtype=torch.float32)
    exponents=((torch.arange(n)[:,None]+torch.arange(k)[None,:]//32)%7)-5
    w=(w*torch.pow(2.0,exponents)).to(torch.bfloat16).to(device)
    p,s=quant_rows(w)
    codes=unpack_codes(p.cpu().numpy()).T.copy()
    nz=torch.from_numpy(pack_nz_from_codes(codes)).to(device)
    scale_pair=torch.from_numpy(prolog_to_matmul_scale(s.view(torch.uint8).cpu().numpy())).to(device).view(torch.float8_e8m0fnu)
    return {'nz':nz,'scale':s.contiguous(),'native':p.T,'native_scale':scale_pair,'row_packed':p}


def mm(p,s,w):
    return torch_npu.npu_quant_matmul(p,w['native'],w['native_scale'],
        pertoken_scale=s.reshape(s.shape[0],-1,2),
        scale_dtype=torch_npu.float8_e8m0fnu,pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        x1_dtype=torch_npu.float4_e2m1fn_x2,x2_dtype=torch_npu.float4_e2m1fn_x2,
        output_dtype=torch.bfloat16,group_sizes=[1,1,32])


def rms(x,gamma,cast_bf16=True):
    # Native MLA RMSNorm: float32 arithmetic and BF16 output before Q re-quant.
    if cast_bf16:
        return torch_npu.npu_rms_norm(x.contiguous(),gamma,epsilon=1e-5)[0]
    xf=x.float();out=xf*torch.rsqrt((xf*xf).mean(-1,keepdim=True)+1e-5)*gamma.float()
    return out.to(torch.bfloat16) if cast_bf16 else out


def rope(x,cos,sin):
    # Prolog RopeVFImpl accepts signed half-split sin; raw weight columns stay
    # interleaved. Output matches GLM interleave after even/odd->half permutation.
    c=cos.float();s=sin.float()
    while c.ndim<x.ndim:c=c.unsqueeze(1);s=s.unsqueeze(1)
    e,o=x.float()[...,::2],x.float()[...,1::2];h=x.shape[-1]//2
    return torch.cat((e*c[...,:h]+o*s[...,:h],o*c[...,h:]+e*s[...,h:]),-1).to(torch.bfloat16)


def native(data,kv_mode):
    p,s=data['p'],data['scale']
    # VA native uses one fused QKV down-projection, then splits Q/KV LoRA.
    down=mm(p,s,data['fused_down'])
    cq=rms(down[:,:2048],data['gamma_q'])
    qp,qs=quant_rows(cq)
    u=mm(qp,qs,data['uq']).reshape(p.shape[0],data['heads'],256)
    q=torch.bmm(u[...,:192].transpose(0,1),data['uk']).transpose(0,1)
    qr=rope(u[...,192:],data['cos'],data['sin'])
    kv=down[:,2048:]
    # New mode preserves VA native BF16 RMSNorm output before C8 cache quant.
    # This deliberately differs from the old Prolog MXFP8 FP32->C8 boundary.
    k=rms(kv[:,:512],data['gamma_k'])
    kr=rope(kv[:,512:],data['cos'],data['sin'])
    if kv_mode==0:return q,qr,cq,k,kr,None
    tiles=k.float().reshape(-1,4,128)
    scales=tiles.abs().amax(-1).clamp_min(1e-4)/448.0
    kq=(tiles/scales[...,None]).to(torch.float8_e4m3fn).reshape(-1,512)
    return q,qr,cq,kq,kr,scales


def native_with_scatter(data,mode,cache,kr,valid,x=None):
    if x is not None:
        px,sx=quant_rows(x);data=dict(data,p=px,scale=sx)
    out=native(data,mode)
    if mode==0:
        cache.reshape(-1,512)[:valid].copy_(out[3][:valid])
        kr.reshape(-1,64)[:valid].copy_(out[4][:valid])
    else:
        raw=cache.view(torch.uint8).reshape(-1,656)[:valid]
        raw[:,:512].copy_(out[3][:valid].contiguous().view(torch.uint8))
        raw[:,512:640].copy_(out[4][:valid].contiguous().view(torch.uint8))
        raw[:,640:].copy_(out[5][:valid].contiguous().view(torch.uint8))
    return out


def metric(name,actual,expected,rtol=.015625,atol=.015625,ratio=.99):
    a=actual.detach().float().cpu().numpy();e=expected.detach().float().cpu().numpy()
    if a.shape!=e.shape:raise AssertionError(f'{name} shape {a.shape} != {e.shape}')
    finite=np.isfinite(a)&np.isfinite(e)
    if not finite.all():raise AssertionError(f'{name}: nonfinite output on finite test inputs')
    err=np.abs(a-e);matched=np.mean(err<=atol+rtol*np.abs(e))
    record={'tensor':name,'matched_ratio':float(matched),'max_abs':float(err.max(initial=0)),
            'rmse':float(np.sqrt(np.mean(err**2))),'rtol':rtol,'atol':atol}
    print('ACCURACY',json.dumps(record),flush=True)
    ACCURACY_RECORDS.append(dict(CASE_CONTEXT,**record))
    if matched<ratio or err.max(initial=0)>1.0:raise AssertionError(f'{name} exceeds provisional tolerance')
    return record


def latency(fn,warmup,iters):
    for _ in range(warmup):fn()
    torch.npu.synchronize();samples=[]
    for _ in range(iters):
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
    return {'p50_ms':float(np.percentile(samples,50)),'p90_ms':float(np.percentile(samples,90)),
            'mean_ms':float(np.mean(samples))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tokens',type=int,nargs='+',default=[1,17])
    parser.add_argument('--heads',type=int,nargs='+',default=[4,8]);parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=20);parser.add_argument('--device',default='npu:0')
    parser.add_argument('--output',default='mla_mxfp4_a5_results.json');args=parser.parse_args()
    if not torch.npu.is_available():raise RuntimeError('NPU unavailable; this test never falls back to CPU')
    if any(t<1 or t>128 for t in args.tokens):raise ValueError('mode6 scope T=1..128')
    torch.npu.set_device(args.device)
    repo=pathlib.Path(__file__).resolve().parents[3]
    ref=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    print(json.dumps({'commit':ref,'api':'custom aclnnMlaPrologV3WeightNz','weight_quant_mode':6,
        'torch':torch.__version__,'torch_npu':torch_npu.__version__,'extension':cann_ops_transformer.__file__,
        'device':torch.npu.get_device_name(),'note':'experimental tolerances; no model-level accuracy claim'}),flush=True)
    gen=torch.Generator().manual_seed(20260914);results=[]
    for heads in args.heads:
        ws={name:weight(k,n,gen,args.device) for name,k,n in [('dq',6144,2048),('uq',2048,heads*256),('dkv',6144,576)]}
        # Concatenate in output-channel order before the one native QKV matmul.
        ws['fused_down']={'native':torch.cat((ws['dq']['row_packed'],ws['dkv']['row_packed']),dim=0).T,
                          'native_scale':torch.cat((ws['dq']['native_scale'],ws['dkv']['native_scale']),dim=1).contiguous()}
        uk=(torch.randn(heads,192,512,generator=gen)/np.sqrt(192)).to(torch.bfloat16).to(args.device)
        for tokens in args.tokens:
            CASE_CONTEXT.clear();CASE_CONTEXT.update(T=tokens,heads=heads)
            x=(torch.randn(tokens,6144,generator=gen)*.1).to(torch.bfloat16).to(args.device)
            p,s=quant_rows(x)
            angle=(torch.arange(tokens)[:,None]+1)*torch.exp(-torch.arange(32)[None,:]/8)
            cos=torch.cat((angle.cos(),angle.cos()),-1).to(torch.bfloat16).to(args.device)
            sin=torch.cat((-angle.sin(),angle.sin()),-1).to(torch.bfloat16).to(args.device)
            data=dict(ws,p=p,scale=s,uk=uk,heads=heads,cos=cos,sin=sin,
                gamma_q=(torch.rand(2048,generator=gen)+.5).to(torch.bfloat16).to(args.device),
                gamma_k=(torch.rand(512,generator=gen)+.5).to(torch.bfloat16).to(args.device))
            # Independent decoded contraction catches native weight/scale producer mistakes.
            aq=dequant_rows(p.cpu().numpy(),s.view(torch.uint8).cpu().numpy())
            wq=dequant_rows(ws['dq']['row_packed'].cpu().numpy(),ws['dq']['scale'].view(torch.uint8).cpu().numpy())[:8]
            decoded=torch.from_numpy((aq@wq.T).astype(np.float32)).to(torch.bfloat16)
            metric('native_dq_vs_decoded',mm(p,s,ws['dq'])[:,:8],decoded)
            pages=(tokens+15)//16+1
            slots=torch.arange(tokens,dtype=torch.int64)
            if tokens>1:slots[-1]=-1
            slots=slots.to(args.device)
            for mode in (0,3):
                baseline=native(data,mode)
                last=None
                for flag in (False,True):
                    CASE_CONTEXT.update(kv_mode=mode,query_norm=flag)
                    dtype=torch.bfloat16 if mode==0 else torch.float8_e4m3fn
                    cache=torch.empty(pages,16,1,512 if mode==0 else 656,dtype=dtype,device=args.device)
                    cache.view(torch.uint8).fill_(0x35)
                    kr=torch.empty(pages,16,1,64,dtype=torch.bfloat16,device=args.device) if mode==0 else torch.empty(0,dtype=torch.bfloat16,device=args.device)
                    kr.view(torch.uint8).fill_(0x35)
                    before=cache.view(torch.uint8).clone();kr_before=kr.view(torch.uint8).clone()
                    def fused(include_quant=False):
                        px,sx=quant_rows(x) if include_quant else (p,s)
                        return cann_ops_transformer.ops.mla_prolog(px,ws['dq']['nz'],ws['uq']['nz'],uk,ws['dkv']['nz'],
                            data['gamma_q'],data['gamma_k'],cache,kr,rope_sin=sin,rope_cos=cos,cache_index=slots,
                            dequant_scale_x=sx,dequant_scale_w_dq=ws['dq']['scale'],dequant_scale_w_uq_qr=ws['uq']['scale'],
                            dequant_scale_w_dkv_kr=ws['dkv']['scale'],weight_quant_mode=6,kv_cache_quant_mode=mode,
                            query_norm_flag=flag,ckvkr_repo_mode=int(mode==3),quant_scale_repo_mode=int(mode==3),tile_size=128)
                    out=fused();torch.npu.synchronize()
                    if out[0].dtype!=torch.bfloat16 or out[1].dtype!=torch.bfloat16:
                        raise AssertionError('query/query_rope must be BF16')
                    if out[2].numel()!=0 or out[2].dtype!=torch.float32:
                        raise AssertionError('query mode0 requires empty FLOAT dequant_scale_q_nope')
                    metric('query',out[0],baseline[0]);metric('query_rope_halfsplit',out[1],baseline[1])
                    if flag:metric('query_norm_bf16_prequant',out[3],baseline[2])
                    elif out[3].numel()!=0:raise AssertionError('query_norm flag=false must return empty')
                    if out[3].dtype!=torch.bfloat16 or out[4].numel()!=0 or out[4].dtype!=torch.float32:
                        raise AssertionError('query_norm dtype/empty FLOAT scale contract')
                    if last is not None:metric('flag_invariance',out[0],last,rtol=0,atol=0,ratio=1)
                    last=out[0].clone()
                    valid=tokens-1 if tokens>1 else tokens
                    rows=cache.reshape(-1,cache.shape[-1])[:valid]
                    if mode==0:
                        metric('kv_bf16',rows,baseline[3][:valid]);metric('kr_bf16',kr.reshape(-1,64)[:valid],baseline[4][:valid])
                    else:
                        raw=rows.view(torch.uint8)
                        fp8=raw[:,:512].contiguous().view(torch.float8_e4m3fn)
                        actual_scale=raw[:,640:656].contiguous().view(torch.float32)
                        raw_diff=float((raw[:,:512]!=baseline[3][:valid].contiguous().view(torch.uint8)).float().mean().item())
                        code_record=dict(CASE_CONTEXT,tensor='kv_fp8_raw_codes',mismatch_ratio=raw_diff,
                                         note='diagnostic only; acceptance compares dequantized values and scales')
                        ACCURACY_RECORDS.append(code_record);print('ACCURACY',json.dumps(code_record),flush=True)
                        decoded=(fp8.float().reshape(-1,4,128)*actual_scale[...,None]).reshape(-1,512)
                        decoded_ref=(baseline[3][:valid].float().reshape(-1,4,128)*baseline[5][:valid,...,None]).reshape(-1,512)
                        metric('kv_fp8_dequantized',decoded,decoded_ref,rtol=.25,atol=.0625)
                        metric('kr_packed_bf16',raw[:,512:640].contiguous().view(torch.bfloat16),baseline[4][:valid])
                        metric('kv_scale_fp32',raw[:,640:656].contiguous().view(torch.float32),baseline[5][:valid],rtol=.015625,atol=1e-6)
                    if not torch.equal(cache.view(torch.uint8).reshape(pages*16,-1)[valid:],before.reshape(pages*16,-1)[valid:]):
                        raise AssertionError('untouched or -1 sentinel cache slot was modified')
                    if mode==0 and not torch.equal(kr.view(torch.uint8).reshape(pages*16,-1)[valid:],kr_before.reshape(pages*16,-1)[valid:]):
                        raise AssertionError('untouched kr cache was modified')
                    perf=latency(fused,args.warmup,args.iterations)
                    full=latency(lambda:fused(True),args.warmup,args.iterations)
                    native_perf=latency(lambda:native(data,mode),args.warmup,args.iterations)
                    native_full=latency(lambda:native_with_scatter(data,mode,cache,kr,valid,x),args.warmup,args.iterations)
                    row={'T':tokens,'heads':heads,'kv_mode':mode,'query_norm':flag,'fused_prequantized':perf,
                         'fused_including_input_quant':full,'native_compute_excluding_cache_scatter':native_perf,
                         'native_including_quant_cache_copy':native_full,
                         'comparison_note':'Full native includes quant and contiguous-slot cache copy; this is a microbenchmark, not arbitrary-page VA throughput.',
                         'full_p50_native_over_fused':native_full['p50_ms']/full['p50_ms']}
                    print('PERF',json.dumps(row),flush=True);results.append(row)
    pathlib.Path(args.output).write_text(json.dumps({'commit':ref,'status':'PASS','accuracy':ACCURACY_RECORDS,'cases':results},indent=2))
    print('MXFP4 MlaPrologV3 A5 accuracy and performance tests passed.',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        pathlib.Path('mla_mxfp4_a5_failure.json').write_text(json.dumps(
            {'status':'FAIL','error':repr(exc),'case':CASE_CONTEXT,'accuracy':ACCURACY_RECORDS},indent=2))
        raise
