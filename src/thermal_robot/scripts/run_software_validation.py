#!/usr/bin/env python3
"""Run a reduced software validation case, preserving logs and process ownership."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sensor-model',choices=['a','b'],default='a')
    parser.add_argument('--strategy',choices=['full','frontier','levy','residual','fast','dual','gp_ucb'],default='dual')
    parser.add_argument('--belief-mode',choices=['off','shadow','online'],default='online')
    parser.add_argument('--duration',type=float,default=60.)
    parser.add_argument('--warmup',type=float,default=36.)
    parser.add_argument('--domain',type=int,default=151)
    parser.add_argument('--scenario',default='static_two_sources.yaml')
    parser.add_argument('--world',default='thermal_scene_nav.world')
    parser.add_argument('--software-params',default='')
    parser.add_argument('--expected-health',default='ready')
    parser.add_argument('--seed',type=int,default=101)
    args=parser.parse_args()
    from run_multiscenario_matrix import runtime_sync_pairs,runtime_sync_mismatches
    mismatches=runtime_sync_mismatches(runtime_sync_pairs())
    if mismatches:
        raise RuntimeError('Rebuild before validation; source/install differ: '+
                           ', '.join(str(src) for src,_ in mismatches))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ROS_DOMAIN_ID=str(args.domain),ROS_LOG_DIR=str(out/'roslog'),
             GAZEBO_LOG_PATH=str(out/'gazebo_log'),GAZEBO_MODEL_DATABASE_URI='',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    bringup=ROOT/'src/thermal_robot/thermal_bringup'
    command=['ros2','launch','thermal_bringup','sim_nav_slam_launch.py','use_rviz:=false','use_gzclient:=false',
             'strategy:='+args.strategy,'sensor_model:='+args.sensor_model,'belief_mode:='+args.belief_mode,
             'run_seed:='+str(args.seed),'scenario_file:='+str(bringup/'config/scenarios'/args.scenario),
             'world_file:='+str(bringup/'worlds'/args.world)]
    if args.software_params:command+=['software_params:='+str(Path(args.software_params).resolve())]
    files=sorted(p for p in (ROOT/'src/thermal_robot').rglob('*') if p.suffix in
                 ('.py','.yaml','.msg','.srv','.world','.urdf','.xml'))
    if args.software_params:files.append(Path(args.software_params).resolve())
    fingerprint=hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in files)).hexdigest()
    (out/'run.json').write_text(json.dumps(dict(command=command,runtime_sha256=fingerprint,args=vars(args)),default=str,indent=2))
    child=None;probe=None;code=2
    # Gazebo Classic uses a process-global master port. Serialize this runner's
    # cases rather than letting separate ROS domains share a physical world.
    lock=open('/tmp/thermal_software_gazebo.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another software validation case is running; retry after it exits')
    try:
        with (out/'launch.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(f'launch pid={child.pid} out={out}',flush=True)
            deadline=time.monotonic()+args.warmup
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError(f'launch exited {child.returncode}')
                if 'process has died' in (out/'launch.log').read_text():
                    raise RuntimeError('A launched node died during warmup; see launch.log')
                time.sleep(.5)
            with (out/'probe.log').open('w') as probe_log:
                probe=subprocess.Popen([sys.executable,str(Path(__file__).with_name('probe_software_stack.py')),
                    '--out',str(out),'--duration',str(args.duration),'--sensor-model',args.sensor_model,
                    '--belief-mode',args.belief_mode,'--expected-health',args.expected_health],cwd=ROOT,env=env,stdout=probe_log,stderr=subprocess.STDOUT,
                    start_new_session=True)
                while probe.poll() is None:
                    if child.poll() is not None:raise RuntimeError('launch stopped during probe')
                    if 'process has died' in (out/'launch.log').read_text():
                        raise RuntimeError('A launched node died during probe; see launch.log')
                    time.sleep(.5)
                code=probe.returncode
            log.flush()
            runtime_log=(out/'launch.log').read_text()
            # Inspect before shutdown: expected SIGINT teardown is not a runtime failure.
            crashes=[line for line in runtime_log.splitlines()
                     if 'Traceback (most recent call last)' in line or 'process has died' in line]
            plan_ms=[float(v) for v in re.findall(r'\[RESIDUAL_PLAN\].*eval_ms=([\d.]+)',runtime_log)]
            diagnostics=dict(crashes=crashes,
                slow_active_reports=runtime_log.count('belief=active'),
                fast_only_reports=runtime_log.count('belief=fast_only'),
                navigation_stall_fallbacks=runtime_log.count('[NAV2_STALL/'),
                source_approach_confirmations=runtime_log.count('[SURFACE_CONFIRMED]'),
                planner_ms_max=max(plan_ms) if plan_ms else None)
            (out/'runtime_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
            if crashes:code=2
    finally:
        for process in (probe,child):
            if process is not None:
                try:process.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        # Gazebo may remain in the original session even when launch exits first.
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        lock.close()
    print(f'probe_exit={code}',flush=True)
    if (out/'probe.json').exists():print((out/'probe.json').read_text(),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
