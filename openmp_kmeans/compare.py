"""Small end-to-end correctness comparisons; run inside an allocated job."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent

def run(command, env=None):
    timeout = int((env if env is not None else os.environ).get('KMEANS_RUN_TIMEOUT', '600'))
    p = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                       text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {command}\n{p.stderr}\n{p.stdout}")
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one CSV result: {p.stdout}")
    centers = [float(v) for line in p.stderr.splitlines() if line.startswith("Centroid ")
               for v in line.split(":", 1)[1].split()]
    return rows[0], centers

def compare(reference, candidate, d, k):
    a, ac = reference
    b, bc = candidate
    if any(a[field] != b[field] for field in ['N','K','D']):
        raise AssertionError("Problem dimensions differ")
    if a['iterations'] != b['iterations']:
        raise AssertionError("Iteration counts differ")
    if len(ac) != d*k or len(bc) != d*k:
        raise AssertionError("Missing centroid output")
    pairs = list(zip(ac, bc)) + [(float(a['wcss']), float(b['wcss']))]
    if not all(math.isfinite(x) and math.isfinite(y) and
               math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-8) for x, y in pairs):
        raise AssertionError("Centroid / WCSS mismatch (rtol=1e-9, atol=1e-8)")
    return {'max_centroid_abs_error': max(abs(x-y) for x,y in zip(ac,bc)),
            'wcss_abs_error': abs(float(a['wcss'])-float(b['wcss']))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['hybrid', 'gpu'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    binaries = ROOT / ("cpu" if args.mode == "hybrid" else "gpu") / "bin"
    env = os.environ.copy()
    env['OMP_DYNAMIC'] = 'FALSE'
    records = []
    # Uneven row partition, zero-row ranks (N=3,P=4), different seeds/dimensions.
    for n,d,k,seed in [(17,2,3,42), (3,1,2,7), (120,8,10,42)]:
        for iterations,tol in [(50,0), (100,1e-6)]:
            params = list(map(str, [n,d,k,seed,iterations,tol])) + ['--centroids']
            serial = 'serial_reference' if args.mode == 'hybrid' else 'serial_gpu_reference'
            ref = run([binaries/serial, *params], env)
            case = dict(N=n,D=d,K=k,seed=seed,max_iter=iterations,tolerance=tol,
                        iterations=int(ref[0]['iterations']))
            if args.mode == 'hybrid':
                for ranks in [1,3,4]:
                    launch = shlex.split(env.get('MPIEXEC', 'mpirun')) + ['-np',str(ranks)]
                    pure = run([*launch, binaries/'mpi_reference', *params], env)
                    records.append(dict(case, ranks=ranks, threads=1, version='mpi',
                                        **compare(ref,pure,d,k)))
                    for threads in [1,2,4]:
                        env['OMP_NUM_THREADS'] = str(threads)
                        got = run([*launch, binaries/'mpi_openmp', *params], env)
                        if int(got[0]['threads']) != threads:
                            raise AssertionError('Requested OpenMP team size was not used')
                        records.append(dict(case, ranks=ranks, threads=threads, version='hybrid',
                                            **compare(ref,got,d,k)))
            else:
                env['OMP_TARGET_OFFLOAD'] = 'MANDATORY'
                for cpu in [True,False]:
                    got = run([binaries/'gpu_openmp', *params, *(['--cpu'] if cpu else [])],env)
                    if got[0]['offloaded'] != ('0' if cpu else '1'):
                        raise AssertionError('Incorrect offload status')
                    records.append(dict(case, version='cpu_assignment' if cpu else 'gpu',
                                        **compare(ref,got,d,k)))
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'status':'PASS','rtol':1e-9,'atol':1e-8,
                               'comparisons':records}, indent=2)+'\n')
    print(f'PASS: {len(records)} cross-version comparisons; {path}')

if __name__ == '__main__':
    main()
