"""Run reproducible Verilog-2001 component/core/UART integration tests."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .prepare_sim import main as prepare_vectors

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iverilog-dir',type=Path)
    args=parser.parse_args()
    found=shutil.which('iverilog')
    directory=args.iverilog_dir or (Path(found).parent if found else Path(r'C:\Users\ndaqu\fpga\fpga.srcs\sources_1\new\.verification-tools\iverilog\ucrt64\bin'))
    compiler=directory/'iverilog.exe';runtime=directory/'vvp.exe'
    if not compiler.exists() or not runtime.exists():
        parser.error('Provide --iverilog-dir pointing to Icarus Verilog bin (iverilog.exe and vvp.exe)')
    prepare_vectors()
    out=ROOT/'build';out.mkdir(exist_ok=True)
    for memory in (ROOT.parent/'digit_recognition'/'export').glob('*.mem'):
        shutil.copy2(memory,out/memory.name)
    sources=sorted((ROOT/'rtl').glob('*.v'))
    suites=[('components','tb_components',8),('components_115200','tb_components',868),
            ('core','tb_inference',None),('uart','tb_uart_system',None)]
    summary=[]
    for name,top,cpb in suites:
        start=time.monotonic()
        command=[str(compiler),'-g2001','-s',top,'-o',str(out/f'{name}_sim.vvp')]
        if cpb is not None:command += [f'-P{top}.CPB={cpb}']
        command += [str(p) for p in sources]+[str(ROOT/'sim'/f'{top}.v')]
        compiled=subprocess.run(command,cwd=out,capture_output=True,text=True,timeout=60)
        if compiled.returncode:
            print(compiled.stdout+compiled.stderr);raise SystemExit(f'{name}: compile failed')
        completed=subprocess.run([str(runtime),str(out/f'{name}_sim.vvp')],cwd=out,capture_output=True,text=True,timeout=180)
        log=completed.stdout+completed.stderr
        (out/f'{name}_sim.log').write_text(log,encoding='utf-8')
        print(log,flush=True)
        if completed.returncode or 'PASS:' not in log or 'FATAL' in log:
            raise SystemExit(f'{name}: simulation failed')
        summary.append({'suite':name,'passed':True,'elapsed_seconds':round(time.monotonic()-start,3)})
    (ROOT.parent/'digit_recognition'/'reports'/'rtl_simulation.json').write_text(json.dumps(summary,indent=2))
    print('ALL FOUR RTL SUITES PASSED',flush=True)

if __name__=='__main__':main()
