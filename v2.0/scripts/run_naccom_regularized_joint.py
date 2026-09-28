"""Staged, cruise-balanced joint training for sparse TA/DIC supervision.

Protocol:
  A. SOCAT-only pretraining of the shared backbone and hydrography adapter.
  B. Five cruise-grouped folds select the chemistry-adapter training length.
  C. Final chemistry-adapter fit, then fixed-length low-LR last-block tuning.
The original GLODAP validation split is an untouched internal test and the
2004--2005 split is loaded only after every final checkpoint exists.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/experiments/joint_four_full_20260907"
OUT = ROOT / "outputs/experiments/joint_regularized_20260908"
CACHE = OLD / "joint_cache.npz"
PREPARED = ROOT / "outputs/prepared_naccom.nc"
EMULATOR = ROOT / "outputs/chem/co2sys_emulator_v1.pt"
SEEDS = (100, 101, 102)
TARGETS = ("sss", "fco2", "dic", "ta")
CHEM_LAMBDA = 0.05
CHEM_SCALE = 20.0
PRETRAIN_STEPS = 3000
CV_MAX_STEPS = 2000
FINETUNE_STEPS = 1500

from recad.chem.co2sys_emulator import CO2SYSEmulator, pyco2sys_fco2
from recad.config import Config
from recad.data.grid import DomainGrid
from recad.data.tensorize import TensorShapes
from recad.model.regularized_joint_st import RegularizedJointST
from recad.model.st_transformer import count_parameters


def atomic_json(path: Path, obj: object) -> None:
    tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8"); tmp.replace(path)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str,float|int]:
    ok=np.isfinite(y)&np.isfinite(p); y=y[ok].astype(float); p=p[ok].astype(float); e=p-y
    return {"n":len(y),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(abs(e))),
            "bias":float(e.mean()),"r2":float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),
            "pearson_r":float(np.corrcoef(y,p)[0,1])}


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float=10.0) -> np.ndarray:
    design=np.column_stack((np.ones(len(x)),x.astype(np.float64)))
    reg=np.eye(design.shape[1])*alpha; reg[0,0]=0
    return np.linalg.solve(design.T@design+reg,design.T@y.astype(np.float64))


def ridge_predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (coef[0]+x.astype(np.float64)@coef[1:]).astype(np.float32)


class Data:
    def __init__(self, include_independent: bool=False):
        self.z=np.load(CACHE,allow_pickle=True)
        self.include_independent=include_independent
        self.fmean=float(self.z["target_stats"][1,0]); self.fstd=float(self.z["target_stats"][1,1])
        sx=self.z["socat_train_x"]; sy=self.z["socat_train_sss"]
        sb=sx[:,5]*self.z["feature_stds"][1]+self.z["feature_means"][1]
        self.sscale=float(np.nanstd(sy-sb))
        self.target_std={"ta":float(np.std(self.z["glodap_train_ta"])),
                         "dic":float(np.std(self.z["glodap_train_dic"]))}
        self.context={"token_feats":torch.from_numpy(self.z["token_feats"]).cuda(),
                      "token_valid":torch.from_numpy(self.z["token_valid"]).cuda()}
        import xarray as xr
        with xr.open_dataset(PREPARED) as ds: grid=DomainGrid(ds.lon.values,ds.lat.values,int(ds.attrs["patch_size"]))
        active=self.z["active_tokens"]; lon,lat=grid.patch_centers()
        coord=np.stack((lon[active]/360,(lat[active]+90)/180),-1).astype(np.float32)
        mon=np.stack((np.sin(2*np.pi*np.arange(1,13)/12),np.cos(2*np.pi*np.arange(1,13)/12)),-1).astype(np.float32)
        ny,_,nt=self.z["token_feats"].shape[:3]
        self.context["coord"]=torch.from_numpy(np.broadcast_to(coord,(ny,12,*coord.shape)).copy()).cuda()
        self.context["month"]=torch.from_numpy(np.broadcast_to(mon[None,:,None,:],(ny,12,nt,2)).copy()).cuda()
        self.token_count=nt

    def ctx(self,year:int): return {k:v[year:year+1] for k,v in self.context.items()}

    def fit_baseline(self, indices: np.ndarray) -> dict[str,np.ndarray|float]:
        x=self.z["glodap_train_x"][indices]
        ct=ridge_fit(x,self.z["glodap_train_ta"][indices]); cd=ridge_fit(x,self.z["glodap_train_dic"][indices])
        rt=self.z["glodap_train_ta"][indices]-ridge_predict(ct,x)
        rd=self.z["glodap_train_dic"][indices]-ridge_predict(cd,x)
        return {"ta_coef":ct,"dic_coef":cd,"ta_scale":float(np.std(rt)),"dic_scale":float(np.std(rd))}

    def records(self,source:str,split:str,baseline:dict,indices:np.ndarray|None=None):
        if split=="independent" and not self.include_independent: raise RuntimeError("independent labels are sealed")
        prefix=f"{source}_{split}"; x=self.z[f"{prefix}_x"]
        if indices is None: indices=np.arange(len(x))
        x=x[indices]; years=self.z[f"{prefix}_year"][indices]
        bsss=x[:,5]*self.z["feature_stds"][1]+self.z["feature_means"][1]
        bta=ridge_predict(baseline["ta_coef"],x); bdic=ridge_predict(baseline["dic_coef"],x)
        groups={}
        for year in np.unique(years):
            ix=np.flatnonzero(years==year); g={
                "x":torch.from_numpy(x[ix]).cuda(),
                "month":torch.from_numpy(self.z[f"{prefix}_month"][indices][ix].astype(np.int64)).cuda(),
                "patch":torch.from_numpy(self.z[f"{prefix}_patch"][indices][ix].astype(np.int64)).cuda(),
                "base_sss":torch.from_numpy(bsss[ix]).cuda(),"base_ta":torch.from_numpy(bta[ix]).cuda(),
                "base_dic":torch.from_numpy(bdic[ix]).cuda()}
            for t in TARGETS:
                key=f"{prefix}_{t}"
                if key in self.z.files:g[t]=torch.from_numpy(self.z[key][indices][ix].astype(np.float32)).cuda()
            if source=="glodap":g["cruise"]=self.z[f"{prefix}_cruise"][indices][ix]
            groups[int(year)]=g
        return groups


def make_model(data:Data,seed:int):
    torch.manual_seed(seed); cfg=Config.from_yaml(ROOT/"configs/experiment_naccom_st_full.yaml").model
    cfg=replace(cfg,embed_dim=192,head="mean",dropout=.1)
    return RegularizedJointST(TensorShapes(data.token_count,1,14,18,7),cfg,chem_width=32).cuda()


def physical(o,g,data,baseline):
    return {"sss":g["base_sss"]+o["sss"].float()*data.sscale,
            "fco2":o["fco2"].float()*data.fstd+data.fmean,
            "ta":g["base_ta"]+o["ta"].float()*float(baseline["ta_scale"]),
            "dic":g["base_dic"]+o["dic"].float()*float(baseline["dic_scale"])}


def sample_group(groups,rng,cruise_balanced=False,batch=512):
    years=sorted(groups); year=int(rng.choice(years)); g=groups[year]
    if cruise_balanced:
        cruises=np.unique(g["cruise"]); picked=rng.choice(cruises,size=batch,replace=True)
        idx=np.asarray([rng.choice(np.flatnonzero(g["cruise"]==c)) for c in picked])
    else: idx=rng.integers(0,len(g["x"]),batch)
    return year,g,torch.from_numpy(idx).cuda()


def direct_loss(o,p,g,idx,source,data,baseline):
    if source=="socat":
        lf=(o["fco2"].float()-(g["fco2"][idx]-data.fmean)/data.fstd).square().mean()
        valid=torch.isfinite(g["sss"][idx]); target=(g["sss"][idx]-g["base_sss"][idx])/data.sscale
        return lf+(o["sss"].float()[valid]-target[valid]).square().mean()
    ta=(g["ta"][idx]-g["base_ta"][idx])/float(baseline["ta_scale"])
    dic=(g["dic"][idx]-g["base_dic"][idx])/float(baseline["dic_scale"])
    return (o["ta"].float()-ta).square().mean()+(o["dic"].float()-dic).square().mean()


def chem_loss(p,g,idx,data,emulator):
    temp=g["x"][idx,4].float()*float(data.z["feature_stds"][0])+float(data.z["feature_means"][0])
    return ((emulator(temp.clamp(-2,35),p["sss"].clamp(5,42),p["ta"].clamp(700,2850),p["dic"].clamp(600,2700))-p["fco2"])/CHEM_SCALE).square().mean()


@torch.inference_mode()
def evaluate(model,data,groups,baseline):
    model.eval(); out={t:{"y":[],"p":[]} for t in TARGETS}
    for year,g in groups.items():
        with torch.autocast("cuda",dtype=torch.bfloat16):ctx=model.encode_context(data.ctx(year))
        for start in range(0,len(g["x"]),8192):
            sl=slice(start,start+8192)
            with torch.autocast("cuda",dtype=torch.bfloat16):o=model.decode_queries(ctx,g["x"][sl],g["month"][sl],g["patch"][sl])
            gg={k:(v[sl] if isinstance(v,torch.Tensor) else v) for k,v in g.items()}; p=physical(o,gg,data,baseline)
            for t in TARGETS:
                if t in g:out[t]["y"].append(g[t][sl].cpu().numpy()); out[t]["p"].append(p[t].cpu().numpy())
    return {t:{k:np.concatenate(v) for k,v in q.items()} for t,q in out.items() if q["y"]}


def pretrain(data,seed,baseline,steps=PRETRAIN_STEPS):
    path=OUT/f"pretrain_seed{seed}.pt"
    if path.exists():return path
    model=make_model(data,seed); groups=data.records("socat","train",baseline)
    # Sparse heads receive no gradient in this phase.
    params=list(model.token_embed.parameters())+list(model.month_proj.parameters())+list(model.spatial.parameters())+list(model.temporal.parameters())+list(model.cell_embed.parameters())+list(model.head.hydro_adapter.parameters())+list(model.head.sss.parameters())+list(model.head.fco2.parameters())
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4); rng=np.random.default_rng(seed)
    for step in range(steps):
        year,g,idx=sample_group(groups,rng); scale=min((step+1)/100,1)*(1 if step<2000 else .3); opt.param_groups[0]["lr"]=1e-3*scale
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):ctx=model.encode_context(data.ctx(year)); o=model.decode_queries(ctx,g["x"][idx],g["month"][idx],g["patch"][idx])
        p=physical(o,{k:(v[idx] if isinstance(v,torch.Tensor) and len(v)==len(g["x"]) else v) for k,v in g.items()},data,baseline)
        loss=direct_loss(o,p,g,idx,"socat",data,baseline); loss.backward(); nn.utils.clip_grad_norm_(params,1); opt.step()
        if (step+1)%500==0:print(f"PRETRAIN seed={seed} step={step+1} loss={loss.item():.4f}",flush=True)
    torch.save({"model":model.state_dict(),"steps":steps},path); del model; torch.cuda.empty_cache(); return path


def train_chem(model,data,groups,baseline,steps,seed,select_groups=None):
    model.freeze_for_chemistry_adapter(); params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=4e-4,weight_decay=3e-3); emulator=CO2SYSEmulator.load_frozen(EMULATOR,"cuda"); rng=np.random.default_rng(seed)
    best=float("inf"); best_state=None; best_step=steps
    for step in range(steps):
        year,g,idx=sample_group(groups,rng,True,256); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):ctx=model.encode_context(data.ctx(year)); o=model.decode_queries(ctx,g["x"][idx],g["month"][idx],g["patch"][idx])
        gi={k:(v[idx] if isinstance(v,torch.Tensor) and len(v)==len(g["x"]) else v) for k,v in g.items()}; p=physical(o,gi,data,baseline)
        loss=direct_loss(o,p,g,idx,"glodap",data,baseline)+CHEM_LAMBDA*chem_loss(p,g,idx,data,emulator)
        loss.backward(); nn.utils.clip_grad_norm_(params,1); opt.step()
        if select_groups is not None and (step+1)%250==0:
            q=evaluate(model,data,select_groups,baseline); score=np.mean([metric(q[t]["y"],q[t]["p"])["rmse"]/data.target_std[t] for t in ("ta","dic")])
            print(f"CV step={step+1} score={score:.4f}",flush=True)
            if score<best:best=score;best_step=step+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:model.load_state_dict(best_state)
    return model,best_step,best


def cruise_folds(cruises,n=5,seed=808):
    unique=np.unique(cruises); rng=np.random.default_rng(seed); rng.shuffle(unique)
    return [np.flatnonzero(np.isin(cruises,x)) for x in np.array_split(unique,n)]


def cross_validate(data,pretrained):
    path=OUT/"cv_results.json"
    if path.exists():return json.loads(path.read_text())
    cruises=data.z["glodap_train_cruise"]; folds=cruise_folds(cruises); rows=[]; allidx=np.arange(len(cruises))
    for k,val in enumerate(folds):
        train=np.setdiff1d(allidx,val); baseline=data.fit_baseline(train); model=make_model(data,100); model.load_state_dict(torch.load(pretrained,map_location="cpu",weights_only=False)["model"])
        tg=data.records("glodap","train",baseline,train); vg=data.records("glodap","train",baseline,val)
        model,step,score=train_chem(model,data,tg,baseline,CV_MAX_STEPS,900+k,vg)
        q=evaluate(model,data,vg,baseline); row={"fold":k,"train_cruises":int(len(np.unique(cruises[train]))),"val_cruises":int(len(np.unique(cruises[val]))),"best_step":step,"criterion":score}
        for t in ("ta","dic"):row[t]=metric(q[t]["y"],q[t]["p"])
        rows.append(row); print("FOLD",k,row,flush=True); del model; torch.cuda.empty_cache()
    atomic_json(path,rows); return rows


def final_fit(data,seed,pretrained,baseline,chem_steps):
    path=OUT/f"final_seed{seed}.pt"
    if path.exists():return path
    model=make_model(data,seed); model.load_state_dict(torch.load(pretrained,map_location="cpu",weights_only=False)["model"])
    gg=data.records("glodap","train",baseline); model,_,_=train_chem(model,data,gg,baseline,chem_steps,seed+1000)
    model.unfreeze_last_blocks(); head=[p for p in model.head.parameters() if p.requires_grad]; backbone=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    opt=torch.optim.AdamW([{"params":head,"lr":3e-4,"weight_decay":1e-3},{"params":backbone,"lr":1e-5,"weight_decay":1e-4}]); emulator=CO2SYSEmulator.load_frozen(EMULATOR,"cuda"); rng=np.random.default_rng(seed+2000); sg=data.records("socat","train",baseline)
    for step in range(FINETUNE_STEPS):
        source="socat" if step%2==0 else "glodap"; groups=sg if source=="socat" else gg; year,g,idx=sample_group(groups,rng,source=="glodap",256)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):ctx=model.encode_context(data.ctx(year)); o=model.decode_queries(ctx,g["x"][idx],g["month"][idx],g["patch"][idx])
        gi={k:(v[idx] if isinstance(v,torch.Tensor) and len(v)==len(g["x"]) else v) for k,v in g.items()}; p=physical(o,gi,data,baseline)
        loss=direct_loss(o,p,g,idx,source,data,baseline)+CHEM_LAMBDA*chem_loss(p,g,idx,data,emulator)
        loss.backward(); nn.utils.clip_grad_norm_(head+backbone,1); opt.step()
        if (step+1)%500==0:print(f"FINETUNE seed={seed} step={step+1} loss={loss.item():.4f}",flush=True)
    torch.save({"model":model.state_dict(),"baseline":baseline,"chem_steps":chem_steps,"finetune_steps":FINETUNE_STEPS},path); del model; torch.cuda.empty_cache(); return path


def score_all(final_paths):
    data=Data(include_independent=True); rows=[]; closure=[]; predictions={}
    for split,rawsplit in (("train","train"),("test","dev"),("independent_2004_2005","independent")):
        source_results={t:[] for t in TARGETS}; truth={}
        for seed,path in zip(SEEDS,final_paths):
            state=torch.load(path,map_location="cpu",weights_only=False); baseline=state["baseline"]; model=make_model(data,seed); model.load_state_dict(state["model"])
            for source in ("socat","glodap"):
                q=evaluate(model,data,data.records(source,rawsplit,baseline),baseline)
                for t,pair in q.items():source_results[t].append(pair["p"]);truth[t]=pair["y"]
            del model;torch.cuda.empty_cache()
        for t in TARGETS:
            p=np.mean(np.stack(source_results[t]),0); predictions[(split,t)]=(truth[t],p); rows.append({"split":split,"target":t,"members":3,**metric(truth[t],p)})
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
    # Exact independent closure on each observation support.
    split="independent"; raw=data.z; fm,fs=raw["feature_means"],raw["feature_stds"]
    for source in ("socat","glodap"):
        all_seed={t:[] for t in TARGETS}; x=raw[f"{source}_{split}_x"]
        for seed,path in zip(SEEDS,final_paths):
            state=torch.load(path,map_location="cpu",weights_only=False);base=state["baseline"];model=make_model(data,seed);model.load_state_dict(state["model"])
            qg=data.records(source,split,base); tmp={t:[] for t in TARGETS}
            model.eval()
            for year,g in qg.items():
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):ctx=model.encode_context(data.ctx(year));o=model.decode_queries(ctx,g["x"],g["month"],g["patch"]);p=physical(o,g,data,base)
                for t in TARGETS:tmp[t].append(p[t].cpu().numpy())
            for t in TARGETS:all_seed[t].append(np.concatenate(tmp[t]))
            del model;torch.cuda.empty_cache()
        pred={t:np.mean(np.stack(all_seed[t]),0) for t in TARGETS}; temp=x[:,4]*fs[0]+fm[0]; exact=pyco2sys_fco2(temp,pred["sss"],pred["ta"],pred["dic"]); closure.append({"support":source,**metric(exact,pred["fco2"])})
    pd.DataFrame(closure).to_csv(OUT/"exact_closure.csv",index=False)
    # Four-panel comparison against the previous model.
    old=pd.read_csv(OLD/"independent_metrics.csv"); old=old[(old.scenario=="b2_soft_co2sys")&(~old.target.eq("co2sys_closure"))]
    frame=pd.DataFrame(rows); ind=frame[frame.split=="independent_2004_2005"]
    fig,ax=plt.subplots(2,2,figsize=(11,8)); units={"sss":"PSU","fco2":"µatm","dic":"µmol kg⁻¹","ta":"µmol kg⁻¹"}
    for a,t in zip(ax.ravel(),TARGETS):
        before=float(old[old.target==t].rmse.iloc[0]); after=float(ind[ind.target==t].rmse.iloc[0]); a.bar(["previous joint","regularized"],[before,after],color=["#999999","#1677b8"]);a.set_title(t.upper());a.set_ylabel(f"RMSE ({units[t]})");a.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(OUT/"independent_comparison.png",dpi=180);plt.close(fig)
    atomic_json(OUT/"completed.json",{"status":"complete"})


def main():
    ap=argparse.ArgumentParser();ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/"protocol.json",{"seeds":list(SEEDS),"pretrain_steps":PRETRAIN_STEPS,"cv_folds":5,"cv_max_steps":CV_MAX_STEPS,"finetune_steps":FINETUNE_STEPS,"chem_lambda":CHEM_LAMBDA,"test":"original GLODAP/SOCAT dev, unopened until final checkpoints","independent":"2004-2005 benchmark; previously opened in prior experiment"})
    data=Data(); full=np.arange(len(data.z["glodap_train_ta"])); baseline=data.fit_baseline(full)
    pre=[pretrain(data,s,baseline) for s in SEEDS]
    cv=cross_validate(data,pre[0]); chem_steps=int(np.median([r["best_step"] for r in cv])); print("SELECTED CHEM STEPS",chem_steps,flush=True)
    final=[final_fit(data,s,p,baseline,chem_steps) for s,p in zip(SEEDS,pre)]
    score_all(final)


if __name__=="__main__":main()
