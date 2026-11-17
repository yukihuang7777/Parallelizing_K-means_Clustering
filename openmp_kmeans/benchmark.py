"""Run five repeats, preserving every raw measurement and identifying versions."""
import argparse
import csv
import os
from pathlib import Path
import shlex
import io
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['hybrid','gpu'])
    p.add_argument('--output',required=True)
    p.add_argument('--ranks',type=int,default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--n',type=int,default=8000000)
    a = p.parse_args()
    binaries = ROOT / ("cpu" if a.mode == "hybrid" else "gpu") / "bin"
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(a.threads)
    env['OMP_DYNAMIC'] = 'FALSE'
    params = list(map(str,[a.n,8,10,42,50,0]))
    if a.mode == 'hybrid':
        ppn = int(env['RANKS_PER_NODE'])
        launch = shlex.split(env.get('MPIEXEC','mpirun')) + [
            '-np',str(a.ranks),'-ppn',str(ppn),'-bind-to',f'core:{a.threads}',
            '-map-by',f'core:{a.threads}']
        configs = [('hybrid', [*launch, binaries/'mpi_openmp', *params])]
        if a.threads == 1:
            configs.insert(0, ('mpi', [*launch, binaries/'mpi_reference', *params]))
    else:
        env['OMP_TARGET_OFFLOAD'] = 'MANDATORY'
        configs = [('serial', [binaries/'serial_gpu_reference', *params]),
                   ('cpu_assignment', [binaries/'gpu_openmp', *params, '--cpu']),
                   ('gpu', [binaries/'gpu_openmp', *params])]
    path = Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields = ['version','repeat','job_id','seed','tolerance','N','K','D','ranks',
              'iterations','time_s','comm_max_s','wcss','threads','gpus','offloaded',
              'assignment_s','device_setup_s']
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for repeat in range(1,6):
            for version,cmd in configs:
                row,_ = run(cmd,env)
                if int(row['iterations']) != 50:
                    raise RuntimeError('Benchmark did not execute 50 iterations')
                if version == 'gpu' and row.get('offloaded') != '1':
                    raise RuntimeError('GPU benchmark did not offload')
                row.setdefault('threads',1)
                row.setdefault('gpus',0)
                row.setdefault('offloaded',0)
                if version == 'hybrid' and int(row['threads']) != a.threads:
                    raise RuntimeError('Unexpected OpenMP team size')
                writer.writerow(dict(row,version=version,repeat=repeat,seed=42,tolerance=0,
                                     job_id=env.get('PBS_JOBID','local')))
                f.flush()
                print(f'{version} repeat {repeat}: {row["time_s"]} s',flush=True)

if __name__ == '__main__':
    main()
