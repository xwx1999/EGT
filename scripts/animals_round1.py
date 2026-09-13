"""Auditable revision experiments; never edit submission files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'results' / 'animals_round1'
TRAITS = {'PigHeaT': ['ADG', 'BFT23', 'RT23'], 'BloodLipid': ['HDL-C', 'LDL-C', 'TCHOL', 'TG'], 'PIC': ['t1', 't2', 't3', 't4', 't5']}
SEEDS = [11, 17, 23, 29, 53]
MODELS = ['gblup', 'svr', 'rf', 'xgboost', 'cnn', 'rnn', 'single', 'mtl']


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def finite(data):
    if isinstance(data, dict):
        return {k: finite(v) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        return [finite(v) for v in data]
    if isinstance(data, (float, np.floating)):
        return float(data) if np.isfinite(data) else None
    if isinstance(data, np.integer):
        return int(data)
    return data


def audit():
    """Extract reviewer text and data provenance before any completion claim."""
    source = (ROOT / 'Animals_Round1_Reviewers_Comments_merged.txt').read_text(encoding='utf-8')
    first, second = source.split('Reviewer 2\n', 1)
    r1 = [p.strip() for p in first.split('\n\n') if p.strip()[:2] in [f'{i}.' for i in range(1, 7)]]
    body = second.split('Comments and Suggestions for Authors\n', 1)[1]
    r2 = [p.strip() for p in body.split('\n\n') if p.strip()][1:14]
    assert len(r1) == 6 and len(r2) == 13
    comments = {**{f'R1-{i}': p for i, p in enumerate(r1, 1)}, **{f'R2-{i}': p for i, p in enumerate(r2, 1)}}
    dump(OUT / 'reviewer_comments_index.json', comments)
    rows = []
    for dataset in TRAITS:
        folder = ROOT / 'data' / 'processed' / dataset
        labels = pd.read_csv(folder / 'labels_long.csv', dtype={'ID': str})
        labels = labels[labels.trait.isin(TRAITS[dataset])]
        for (trait, split), frame in labels.groupby(['trait', 'split']):
            y = frame.value.dropna()
            rows.append(dict(dataset=dataset, trait=trait, split=split, n=len(y), mean=y.mean(), sd=y.std(), skewness=y.skew(), zero_fraction=(y == 0).mean(), minimum=y.min(), maximum=y.max(), unique_values=y.nunique(), binary=bool(y.isin([0, 1]).all()), integer_valued=bool(np.allclose(y, np.round(y))), units='not established from released numeric file'))
        for name in ('labels_long.csv', 'splits.csv', 'markers.csv'):
            path = folder / name
            if path.exists():
                dump(OUT / 'provenance' / f'{dataset}_{name}.json', {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        splits = pd.read_csv(folder / 'splits.csv', dtype={'ID': str})
        assert not splits.ID.duplicated().any(), f'Duplicate split assignment: {dataset}'
        if 'population' in splits:
            splits.groupby(['population', 'split']).size().rename('n').to_csv(OUT / f'{dataset}_population_split_counts.csv')
        markers = pd.read_csv(folder / 'markers.csv')
        if 'chromosome' in markers.columns:
            markers.groupby('chromosome').size().rename('n_markers').to_csv(OUT / f'{dataset}_chromosome_counts.csv')
        else:
            dump(OUT / f'{dataset}_chromosome_counts.json', {'status': 'missing_chromosome_annotation', 'columns': list(markers.columns), 'reviewer_mapping': 'R2-9'})
    pd.DataFrame(rows).to_csv(OUT / 'phenotype_distributions.csv', index=False)
    signatures = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'animals_submission_20260826_final').rglob('*') if p.is_file() and p.suffix in ('.tex', '.pdf')}
    target = OUT / 'submission_hashes_before.json'
    if not target.exists():
        dump(target, signatures)
    dump(OUT / 'submission_unchanged.json', {'unchanged': json.loads(target.read_text()) == signatures})


def load_bundle(dataset, trait, test, multitrait=False):
    import baselines as b
    import egt
    folder = ROOT / 'data' / 'processed' / dataset
    kw = dict(genotype_train=folder / 'genotypes_train.csv', genotype_valid=folder / 'genotypes_valid.csv', genotype_test=folder / 'genotypes_test.csv' if test else None, labels_path=folder / 'labels_long.csv', id_col='ID', target_column='value', trait_column='trait', split_column='split')
    if multitrait:
        return egt.load_multitrait_bundle(**kw, traits=TRAITS[dataset])
    return b.load_dataset_bundle(**kw, genotypes=None, trait=trait)


def worker(job_path):
    import torch
    import baselines as b
    import egt
    from dataclasses import asdict
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'DONE.json').exists():
        return
    assert torch.cuda.is_available(), 'GPU training requires CUDA in the existing conda environment'
    torch.set_num_threads(4)
    b.set_random_seed(job['seed'])
    dataset, name, trait = job['dataset'], job['model'], job['trait']
    test = job['phase'] == 'repeat'
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    bundle = load_bundle(dataset, trait, test, name == 'mtl')
    load_seconds = time.perf_counter() - started
    t0 = time.perf_counter()
    if name in ('single', 'mtl'):
        if name == 'single':
            bundle = egt.single_to_multitrait_bundle(bundle)
        from run_all_trait_egt import EXPERIMENTS
        exp = next(x for x in EXPERIMENTS if x['dataset'] == dataset)
        settings = dict(exp['mtl' if name == 'mtl' else 'single'])
        if name == 'single':
            settings.update(exp.get('single_traits', {}).get(trait, {}))
        config = egt.EGTConfig(**settings, seed=job['seed'], device='cuda', num_workers=0)
        model, state = egt.train_egt(bundle, config)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0
        prediction_rows = []
        inference = {}
        for split in (['valid', 'test'] if test else ['valid']):
            data = getattr(bundle, split)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            z = egt.predict_arrays(model, state[f'X_{split}'], config.batch_size, 0, torch.device('cuda'), config.amp)
            torch.cuda.synchronize()
            inference[split] = time.perf_counter() - t1
            pred = z * state['y_stds'] + state['y_means']
            for i, tr in enumerate(bundle.traits):
                mask = data.mask[:, i]
                prediction_rows.append(pd.DataFrame({'ID': data.ids[mask], 'trait': tr, 'split': split, 'y_true': data.y[mask, i], 'y_pred': pred[mask, i]}))
        predictions = pd.concat(prediction_rows, ignore_index=True)
        egt.write_training_history(state, output)
        # Retain lipid checkpoints for later IG/LD controls without saving every model.
        if dataset == 'BloodLipid' and name == 'single' and test:
            egt.save_checkpoint(model, bundle, state, config, output)
        configuration = asdict(config)
        params = sum(p.numel() for p in model.parameters())
    else:
        config = b.TorchTrainingConfig(epochs=100, batch_size=128, device='cuda', seed=job['seed'], selection_metric='pearson')
        if name in ('cnn', 'rnn'):
            model = b.fit_torch_model(name, bundle, config)
            configuration = asdict(config)
        else:
            model, configuration = b.fit_classical_model(name, bundle, job['seed'], 4, 'cuda')
            if hasattr(model, 'get_params'):
                configuration['estimator_params'] = {k: repr(v) for k, v in model.get_params().items()}
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0
        prediction_rows, inference = [], {}
        for split in (['valid', 'test'] if test else ['valid']):
            data = getattr(bundle, split)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            pred = model.predict(data.X)
            torch.cuda.synchronize()
            inference[split] = time.perf_counter() - t1
            prediction_rows.append(pd.DataFrame({'ID': data.ids, 'trait': trait, 'split': split, 'y_true': data.y, 'y_pred': pred}))
        predictions = pd.concat(prediction_rows, ignore_index=True)
        params = None
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics = []
    for (tr, split), group in predictions.groupby(['trait', 'split']):
        y, p = group.y_true.to_numpy(), group.y_pred.to_numpy()
        m = b.regression_metrics(y, p)
        m.update(trait=tr, split=split, bias=float(np.mean(p-y)), slope=float(np.cov(p,y,ddof=0)[0,1]/np.var(p)) if np.var(p)>0 else float('nan'))
        metrics.append(m)
    dump(output / 'metrics.json', finite(metrics))
    dump(output / 'configuration.json', finite(configuration))
    dump(output / 'cost.json', {'load_seconds':load_seconds,'training_seconds_including_ae':train_seconds,'inference_seconds':inference,'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_memory_scope':'PyTorch only; does not capture XGBoost allocation','parameters':params,'gpu':torch.cuda.get_device_name(0),'wall_seconds':time.perf_counter()-started})
    dump(output / 'DONE.json', {'finished':datetime.now().isoformat(), 'job':job})


def jobs(phase):
    rows = []
    for seed in ([42] if phase == 'selection' else SEEDS):
        for dataset, traits in TRAITS.items():
            for name in MODELS:
                for trait in (['all'] if name == 'mtl' else traits):
                    key = f'{dataset}_{trait}_{name}_{seed}'
                    rows.append(dict(key=key, phase=phase, dataset=dataset, trait=trait, model=name, seed=seed, output=str(OUT / phase / key)))
    return rows


def run_queue(rows):
    active = {}
    pending = iter(rows)
    exhausted = False
    completed = 0
    failed = 0
    total = len(rows)
    progress_log = OUT / 'animals_round1_progress.log'
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    def report(message):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with progress_log.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    report(f"QUEUE_START total={total} workers=2 gpus=0,1")
    while active or not exhausted:
        for gpu in range(2):
            if gpu in active:
                continue
            try:
                job = next(pending)
            except StopIteration:
                exhausted = True
                break
            path = OUT / 'jobs' / f"{job['phase']}_{job['key']}.json"
            dump(path, job)
            if (Path(job['output']) / 'DONE.json').exists():
                continue
            log = (OUT / 'jobs' / f"{job['phase']}_{job['key']}.log").open('w', encoding='utf-8')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--job', str(path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (proc, log, job)
            report(f"START gpu={gpu} job={job['key']} progress={completed + failed}/{total} active={len(active)}")
        for gpu, (proc, log, job) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            if code != 0:
                failed += 1
                dump(OUT / 'failures' / f"{job['phase']}_{job['key']}.json", dict(job=job, exit_code=code))
                report(f"FAIL gpu={gpu} job={job['key']} exit={code} progress={completed + failed}/{total} active={len(active)}")
            else:
                completed += 1
                report(f"DONE gpu={gpu} job={job['key']} progress={completed + failed}/{total} active={len(active)}")
        time.sleep(2)
    report(f"QUEUE_END completed={completed} failed={failed} total={total}")


def lock_selection():
    records = []
    for job in jobs('selection'):
        directory = Path(job['output'])
        if not (directory / 'DONE.json').exists():
            raise RuntimeError(f"Selection incomplete: {job['key']}")
        for m in json.loads((directory / 'metrics.json').read_text()):
            assert m['split'] == 'valid'
            records.append(dict(dataset=job['dataset'], model=job['model'], **m))
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'selection_validation_metrics.csv', index=False)
    winners = []
    for (dataset, trait), group in df.groupby(['dataset', 'trait']):
        for family, names in [('EGT', ['single','mtl']), ('baseline', MODELS[:6])]:
            candidates = group[group.model.isin(names)].dropna(subset=['pearson'])
            if candidates.empty:
                raise RuntimeError(f'No finite validation candidate: {dataset}/{trait}/{family}')
            row = candidates.sort_values(['pearson','rmse','model'], ascending=[False,True,True]).iloc[0]
            winners.append(dict(dataset=dataset, trait=trait, family=family, model=row.model, validation_pcc=row.pearson))
    dump(OUT / 'selection_lock.json', {'locked_at':datetime.now().isoformat(),'selection_seed':42,'repeat_seeds':SEEDS,'test_used_for_selection':False,'historical_test_exposure':'Previously inspected; reruns are not an independent prospective test','winners':winners})


def summarize():
    records, costs = [], []
    for job in jobs('repeat'):
        directory = Path(job['output'])
        if not (directory / 'DONE.json').exists():
            continue
        for m in json.loads((directory / 'metrics.json').read_text()):
            records.append(dict(dataset=job['dataset'],model=job['model'],seed=job['seed'],**m))
        costs.append(dict(dataset=job['dataset'],model=job['model'],seed=job['seed'],trait=job['trait'],**json.loads((directory/'cost.json').read_text())))
    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUT/'repeat_metrics_long.csv',index=False)
        df.groupby(['dataset','trait','model','split'])[['pearson','rmse','mae','bias','slope']].agg(['count','mean','std']).to_csv(OUT/'repeat_metrics_summary.csv')
        pd.DataFrame(costs).to_csv(OUT/'computational_cost_partial.csv',index=False)
    dump(OUT/'benchmark_status.json',{'expected_jobs':len(jobs('repeat')),'finished_jobs':len(costs),'complete':len(costs)==len(jobs('repeat'))})


def pipeline():
    audit()
    dump(OUT/'execution_protocol.json',{'created':datetime.now().isoformat(),'selection_jobs':len(jobs('selection')),'repeat_jobs':len(jobs('repeat')),'seeds':SEEDS,'selection':'seed42 validation only; all candidates fixed before new test evaluations','scope':'main benchmark only; ablation repetitions and work packages 5-12 require separate execution','no_test_during_selection':True})
    run_queue(jobs('selection'))
    lock_selection()
    run_queue(jobs('repeat'))
    summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit','pipeline','summarize','worker'])
    parser.add_argument('--job')
    args=parser.parse_args()
    if args.action == 'worker':
        worker(args.job)
    else:
        globals()[args.action]()
