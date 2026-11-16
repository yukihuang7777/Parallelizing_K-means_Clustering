"""Bounded startup and binding checks using a small, uneven dataset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CPU = Path(__file__).resolve().parent
sys.path.insert(0, str(CPU.parent))
from compare import run, compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    env = os.environ.copy()
    hosts = int(env['KMEANS_HOST_COUNT'])
    # 1003 is not divisible by any of 16, 32, 64. Three fixed iterations.
    params = ['1003', '8', '10', '42', '3', '0', '--centroids']
    reference = run([CPU/'bin/serial_reference', *params], env)
    records = []
    for ranks, threads in [(16, 4), (32, 2), (64, 1)]:
        env['OMP_NUM_THREADS'] = str(threads)
        launch = ['mpirun', '-np', str(ranks), '-ppn', str(ranks // hosts),
                  '-bind-to', f'core:{threads}', '-map-by', f'core:{threads}']
        print(f'CHECK {ranks} ranks x {threads} threads: affinity', flush=True)
        affinity = subprocess.run(
            [*launch, 'sh', '-c', 'echo "host=$(hostname) rank=${PMI_RANK:-unknown}"; taskset -pc $$'],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        out = Path(args.output).parent/f'affinity_{ranks}x{threads}.txt'
        out.write_text(affinity.stdout)
        if affinity.returncode:
            raise RuntimeError(f'Affinity launch failed; see {out}\n{affinity.stdout}')
        got = run([*launch, CPU/'bin/mpi_openmp', *params], env)
        if int(got[0]['threads']) != threads or int(got[0]['ranks']) != ranks:
            raise RuntimeError('Unexpected rank/thread count')
        records.append(dict(ranks=ranks, threads=threads, version='hybrid',
                            **compare(reference, got, 8, 10)))
        if threads == 1:
            got = run([*launch, CPU/'bin/mpi_reference', *params], env)
            records.append(dict(ranks=ranks, threads=1, version='mpi',
                                **compare(reference, got, 8, 10)))
        print(f'PASS {ranks}x{threads}', flush=True)
    Path(args.output).write_text(json.dumps({'status': 'PASS', 'checks': records}, indent=2)+'\n')


if __name__ == '__main__':
    main()
