"""Offline publication audit: recompute saved FINAL V2 metrics without training.
New verification utility; reference source/results remain unchanged. Writes only the
explicit --report destination (default work/verification.json).
"""
import argparse, ast, csv, gzip, hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/final_v2'
checks = []


def check(name, condition, detail=None):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        raise AssertionError(name + ': ' + str(detail))


def close(a, b, atol=1e-12):
    return bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))


def metrics(g):
    y = g.label.to_numpy(dtype=int)
    p = g.probability.to_numpy(dtype=float)
    z = g.prediction.to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y,z,labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,z), 'precision':precision_score(y,z,zero_division=0), 'recall':recall_score(y,z,zero_division=0), 'f1':f1_score(y,z,zero_division=0), 'roc_auc':roc_auc_score(y,p), 'pr_auc':average_precision_score(y,p), 'brier_score':brier_score_loss(y,p), 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def compare_frames(actual, expected, keys, columns, name):
    a=actual.set_index(keys).sort_index(); b=expected.set_index(keys).sort_index()
    check(name+' keys', a.index.equals(b.index), [len(a),len(b)])
    for c in columns:
        check(name+' '+c, close(a[c],b[c]), {'maximum_absolute_error':float(np.nanmax(np.abs(a[c].to_numpy(dtype=float)-b[c].to_numpy(dtype=float))))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'work/verification.json')
    args=parser.parse_args()
    # Full payload digests; gzip is also checked against original uncompressed bytes.
    for row in json.loads((ROOT/'provenance/source_artifacts.json').read_text()):
        b=(ROOT/row['published']).read_bytes()
        check('published digest '+row['published'],hashlib.sha256(b).hexdigest()==row['published_sha256'])
        if row['transformation'].startswith('lossless gzip'):
            check('original prediction bytes '+row['published'],hashlib.sha256(gzip.decompress(b)).hexdigest()==row['source_sha256'])
    splits={k:pd.read_csv(ROOT/f'splits/{k}.csv') for k in ['train','validation','test']}
    manifest=pd.read_csv(ROOT/'splits/split_manifest.csv')
    check('split counts',[len(splits[k]) for k in ['train','validation','test']]==[3524,753,751])
    check('all IDs unique',len(manifest)==5028 and manifest.id.is_unique)
    check('split union',set(manifest.id)==set().union(*(set(x.id) for x in splits.values())))
    for name,g in splits.items():
        check('split order '+name,g.equals(manifest[manifest.split==name].reset_index(drop=True)))
    check('leakage groups disjoint',manifest.groupby('leakage_group_id').split.nunique().max()==1)
    counts=manifest.groupby('leakage_group_id').size()
    check('manuscript group statistics', [len(counts),int((counts>1).sum()),int(counts[counts>1].sum()),int(counts.max())]==[4457,228,799,23])
    check('manuscript dataset classes',manifest.label.value_counts().to_dict()=={0.0:2833,1.0:2195})
    check('manuscript platform totals',manifest.platform.value_counts().to_dict()=={'X':2167,'TikTok':1442,'Facebook':1419})
    # Both full ID metadata and numerical predictions are necessary for direct replay.
    test=splits['test'].set_index('id'); valid=splits['validation'].set_index('id')
    pred=pd.read_csv(BASE/'predictions/robustness_all_predictions.csv.gz')
    protocol=json.loads((BASE/'configs/final_training_protocol.json').read_text())
    models=protocol['models']; seeds=protocol['random_seeds']
    conditions=list(json.loads((BASE/'configs/robustness_test_conditions.json').read_text()))
    check('complete experimental grid',len(pred)==289135 and set(pred.model_name)==set(models) and set(pred.seed)==set(seeds) and set(pred.condition)==set(conditions))
    keys=['model_name','seed','condition']
    check('prediction key uniqueness',not pred.duplicated(keys+['id']).any())
    check('prediction metadata',np.array_equal(pred.label.to_numpy(),test.loc[pred.id].label.to_numpy()) and np.array_equal(pred.platform.to_numpy(),test.loc[pred.id].platform.to_numpy()))
    check('probabilities finite and bounded',np.isfinite(pred[['logit','probability','threshold']]).all().all() and pred.probability.between(0,1).all())
    check('sigmoid consistency',close(pred.probability,1/(1+np.exp(-pred.logit)),atol=1e-7))
    check('threshold decisions exact',np.array_equal((pred.probability>=pred.threshold).astype(int),pred.prediction))
    check('correctness flags exact',np.array_equal(pred.is_correct.to_numpy(),(pred.prediction==pred.label).to_numpy()))
    check('thresholds frozen across conditions',(pred.groupby(['model_name','seed']).threshold.nunique()==1).all())
    metric_rows=[]
    for key,g in pred.groupby(keys,sort=True):
        check('test row order '+str(key),g.id.tolist()==splits['test'].id.tolist())
        check('test count '+str(key),len(g)==751)
        metric_rows.append(dict(zip(keys,key))|{'test_'+k if k not in ['tn','fp','fn','tp'] else k:v for k,v in metrics(g).items()})
    calc=pd.DataFrame(metric_rows)
    expected=pd.read_csv(BASE/'summaries/robustness_per_seed_summary.csv')
    metric_cols=['test_'+k for k in ['accuracy','precision','recall','f1','roc_auc','pr_auc','brier_score']]+['tn','fp','fn','tp']
    compare_frames(calc,expected,keys,metric_cols,'385 recomputed metric evaluations')
    clean=calc[calc.condition=='clean'].set_index(['model_name','seed'])
    for m,s in clean.index:
        vr=pd.read_csv(BASE/'summaries'/({'text_only_mlp':'text_only_mlp_training_summary.csv','image_only_mlp':'image_only_mlp_training_summary.csv'}.get(m,m+'_validation_summary.csv')))
        val=float(vr.loc[vr.seed==s,'selected_threshold'].iloc[0])
        actual=pred.loc[(pred.model_name==m)&(pred.seed==s),'threshold'].iloc[0]
        check(f'clean-validation threshold {m}/{s}',abs(val-actual)<1e-12)
    # Available validation predictions exist for the two component MLPs and average fusion only.
    threshold_grid=np.round(np.arange(.05,.951,.01),2)
    for file in sorted((BASE/'predictions').glob('validation_*.csv.gz')):
        g=pd.read_csv(file); m=g.model_name.iloc[0]; s=int(g.seed.iloc[0])
        check('validation IDs '+file.name,g.id.tolist()==splits['validation'].id.tolist())
        check('validation metadata '+file.name,np.array_equal(g.label.to_numpy(),valid.loc[g.id].label.to_numpy()) and np.array_equal(g.platform.to_numpy(),valid.loc[g.id].platform.to_numpy()))
        check('validation probabilities '+file.name,np.isfinite(g.probability).all() and g.probability.between(0,1).all())
        check('validation decisions '+file.name,np.array_equal((g.probability>=g.threshold).astype(int),g.prediction))
        search=[]
        for t in threshold_grid:
            z=(g.probability>=t).astype(int)
            search.append({'threshold':t,'f1':f1_score(g.label,z,zero_division=0),'precision':precision_score(g.label,z,zero_division=0),'recall':recall_score(g.label,z,zero_division=0)})
        best=pd.DataFrame(search).sort_values(['f1','precision','recall','threshold'],ascending=[False,False,False,True]).iloc[0].threshold
        check('validation threshold argmax '+file.name,abs(best-g.threshold.iloc[0])<1e-12)
    agg=calc.groupby(['model_name','condition']).agg(mean_test_f1=('test_f1','mean'),std_test_f1=('test_f1','std'),mean_test_precision=('test_precision','mean'),mean_test_recall=('test_recall','mean'),mean_test_pr_auc=('test_pr_auc','mean'),mean_test_brier_score=('test_brier_score','mean')).reset_index()
    saved_agg=pd.read_csv(BASE/'summaries/robustness_model_condition_aggregate.csv')
    compare_frames(agg,saved_agg,['model_name','condition'],[c for c in agg if c not in ['model_name','condition']],'77 mean and sample-SD aggregates')
    clean_saved=pd.read_csv(BASE/'predictions/clean_test_all_model_predictions.csv.gz')
    pc=pred[pred.condition=='clean'].drop(columns='category').reset_index(drop=True)
    cc=clean_saved.sort_values(keys+['id']).reset_index(drop=True);pc=pc.sort_values(keys+['id']).reset_index(drop=True)
    check('separate clean predictions identical',pc.equals(cc))
    # All platform per-seed F1, precision, recall, AP, Brier and aggregate means/SD.
    pr=[]
    for key,g in pred.groupby(keys+['platform'],sort=True):
        mm=metrics(g)
        pr.append(dict(zip(keys+['platform'],key))|{'samples':len(g),'positive_samples':int(g.label.sum())}|{k:mm[k] for k in ['f1','precision','recall','pr_auc','brier_score']})
    platforms=pd.DataFrame(pr)
    compare_frames(platforms,pd.read_csv(BASE/'per_platform/robustness_platform_per_seed.csv'),keys+['platform'],['samples','positive_samples','f1','precision','recall','pr_auc','brier_score'],'1155 platform evaluations')
    pagg=platforms.groupby(['model_name','condition','platform']).agg(mean_f1=('f1','mean'),std_f1=('f1','std'),mean_precision=('precision','mean'),mean_recall=('recall','mean'),mean_pr_auc=('pr_auc','mean'),mean_brier_score=('brier_score','mean')).reset_index()
    compare_frames(pagg,pd.read_csv(BASE/'per_platform/robustness_platform_aggregate.csv'),['model_name','condition','platform'],[c for c in pagg if c not in ['model_name','condition','platform']],'231 platform aggregates')
    # Transcribed manuscript tables are independent targets.
    targets=json.loads((ROOT/'provenance/manuscript_tables.json').read_text())
    ac=agg[agg.condition=='clean'].set_index('model_name')
    for m,row in targets['table_3'].items():
        for metric,val in row.items():check('manuscript Table 3 '+m+'/'+metric,round(float(ac.loc[m,metric])*100,2)==val)
    aa=agg.set_index(['model_name','condition'])
    for c,row in targets['table_4'].items():
        for m,val in row.items():check('manuscript Table 4 '+m+'/'+c,round(float(aa.loc[(m,c),'mean_test_f1'])*100,2)==val)
    main=pd.read_csv(BASE/'summaries/final_main_robustness_results.csv').set_index('model_name')
    jc=['joint_j1_text_obfuscated_blur_r5','joint_j2_text_obfuscated_pixelation_p12_5','joint_j3_text_obfuscated_blur_r5_pixelation_p12_5']
    for m in models:
        vals=[float(aa.loc[(m,c),'mean_test_f1']) for c in jc];cf=float(ac.loc[m,'mean_test_f1'])
        computed={'clean_f1':cf,'j1_f1':vals[0],'j2_f1':vals[1],'j3_f1':vals[2],'mean_joint_f1':float(np.mean(vals)),'worst_joint_f1':min(vals),'worst_joint_drop':cf-min(vals),'worst_joint_relative_drop_pct':(cf-min(vals))/cf*100}
        for k,v in computed.items():check('joint summary '+m+'/'+k,abs(v-main.loc[m,k])<1e-12)
    for m,row in targets['table_5'].items():
        for k,v in row.items():
            scale=1 if k=='worst_joint_relative_drop_pct' else 100
            check('manuscript Table 5 '+m+'/'+k,round(float(main.loc[m,k])*scale,2)==v)
    bs=pd.read_csv(BASE/'bootstrap/paired_bootstrap_top3_primary_conditions.csv')
    check('12 bootstrap comparisons',len(bs)==12 and (bs.bootstrap_iterations==5000).all())
    check('no statistical superiority',(bs.ci_95_lower<=0).all() and (bs.ci_95_upper>=0).all() and (~bs.significant_95_percent).all() and (bs.approximate_p_value>.05).all())
    check('minimum bootstrap p=0.1016',abs(bs.approximate_p_value.min()-.1016)<1e-12)
    for row in bs.itertuples():
        check('bootstrap observed F1 '+row.condition+'/'+row.model_a,abs(aa.loc[(row.model_a,row.condition),'mean_test_f1']-row.observed_mean_f1_a)<1e-12 and abs(aa.loc[(row.model_b,row.condition),'mean_test_f1']-row.observed_mean_f1_b)<1e-12)
    # Numeric claims in sections 3.2-3.4 and the conclusion.
    check('text-only perturbation claim',round(aa.loc[('text_only_mlp','text_obfuscated'),'mean_test_f1']*100,2)==92.37)
    check('image-only compound claim',round(aa.loc[('image_only_mlp','blur_r5_pixelation_p12_5'),'mean_test_f1']*100,2)==91.74)
    check('TikTok positive test samples',int(test[test.platform=='TikTok'].label.sum())==30 and len(test[test.platform=='TikTok'])==216)
    top=['learnable_weighted_logit_fusion','raw_concatenation_fusion','average_logit_fusion']
    pindex=pagg.set_index(['model_name','condition','platform'])
    for c,v in zip(jc,[55.75,59.47,60.23]):check('TikTok raw concat '+c,round(pindex.loc[('raw_concatenation_fusion',c,'TikTok'),'mean_f1']*100,2)==v)
    tp=pred[(pred.platform=='TikTok')&(pred.label==1)&pred.model_name.isin(top)&pred.condition.isin(['clean']+jc)]
    fn=tp.groupby(['id','model_name','condition']).prediction.apply(lambda z:int((z==0).sum())).reset_index(name='missed_seed_count')
    fn=fn[fn.missed_seed_count>=4]
    check('20 persistent TikTok positives',fn.id.nunique()==20)
    stored_fn=pd.read_csv(BASE/'diagnostics/tiktok_persistent_false_negatives.csv')
    compare_frames(fn,stored_fn,['id','model_name','condition'],['missed_seed_count'],'persistent TikTok false negatives')
    audit=pd.read_csv(BASE/'diagnostics/tiktok_persistent_fn_label_audit.csv')
    check('manual review categories',len(audit)==6 and audit.audit_category.value_counts().to_dict()=={'implicit_gambling_context':2,'discussion_or_complaint':2,'unrelated_content':1,'ambiguous':1})
    check('manual review issue counts',audit.possible_label_issue.value_counts().to_dict()=={'yes':4,'uncertain':2})
    # Generation manifests are identifier-only publication projections or unchanged image metadata.
    text=pd.read_csv(ROOT/'perturbations/final_v2/manifests/text/test_text_obfuscated_fullchar_period.csv')
    check('text obfuscation coverage and order',text.id.tolist()==splits['test'].id.tolist() and int(text.text_changed.sum())==314 and int(text.target_occurrence_count.sum())==4540 and int(text.remaining_target_count.sum())==0)
    image=pd.read_csv(ROOT/'perturbations/final_v2/manifests/image/test_all_image_corruptions.csv')
    check('six corruption manifest sizes',len(image)==4506 and (image.groupby('condition').size()==751).all())
    for c,g in image.groupby('condition'):check('corruption ID order '+c,g.id.tolist()==splits['test'].id.tolist())
    # Parse notebook and scripts, ensuring outputs/attachments and absolute local paths are absent.
    for file in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src').glob('*.py')):
        ast.parse(file.read_text(),filename=str(file))
    for notebook in sorted((ROOT/'notebooks').glob('*.ipynb')):
        nb=json.loads(notebook.read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(c['source']))
                check('clean notebook '+notebook.name+'/'+c['id'],c['outputs']==[] and c['execution_count'] is None and not c['metadata'])
            check('no notebook attachments '+notebook.name+'/'+c['id'],'attachments' not in c)
    patterns=[r'/Users/',r'/home/',r'[A-Za-z]:\\Users\\',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{20,}',r'hf_[A-Za-z0-9]{20,}',r'AKIA[0-9A-Z]{16}',r'(?<![A-Za-z0-9])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}']
    for file in ROOT.rglob('*'):
        if not file.is_file() or 'work' in file.relative_to(ROOT).parts or '__pycache__' in file.parts:continue
        if file.suffix not in ['.py','.ipynb','.json','.md','.txt','.csv']:continue
        content=file.read_text()
        if file.resolve()==Path(__file__).resolve():
            content="\n".join(line for line in content.splitlines() if not line.lstrip().startswith("patterns="))
        for index,pattern in enumerate(patterns):check('sensitive/path scan '+str(file.relative_to(ROOT))+'/pattern_'+str(index),re.search(pattern,content) is None)
    report={'status':'passed','scope':'offline data/schema/integrity/metric/manuscript checks; no encoder inference or training','prediction_rows':len(pred),'metric_evaluations':len(calc),'aggregate_rows':len(agg),'platform_metric_evaluations':len(platforms),'manuscript_tables_verified':[3,4,5],'bootstrap_resampling_rerun':False,'check_count':len(checks),'checks':checks}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':
    main()
