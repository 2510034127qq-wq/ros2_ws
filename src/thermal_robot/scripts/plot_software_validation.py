#!/usr/bin/env python3
"""Plot retained A/B software probe artifacts without rerunning a simulation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.run/'probe.json').read_text())
    with np.load(args.run/'snapshots.npz') as data:
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,key,title,unit in [(axes[0,0],'raw','Thermal observation','Celsius'),
                                  (axes[0,1],'depth','Registered optical depth','m'),
                                  (axes[1,0],'map','Fused ground projection','Celsius')]:
            if key not in data:
                ax.text(.5,.5,'Not used in A-level model',ha='center');ax.set_axis_off();continue
            arr=data[key]
            if key=='map':
                # Crop to finite non-ambient information to keep the view legible.
                hot=np.argwhere(arr>24.)
                if len(hot):
                    lo=np.maximum(hot.min(axis=0)-8,0);hi=np.minimum(hot.max(axis=0)+9,arr.shape)
                    arr=arr[lo[0]:hi[0],lo[1]:hi[1]]
            im=ax.imshow(arr,cmap='inferno' if key!='depth' else 'viridis',origin='upper')
            ax.set_title(title);ax.set_xlabel('pixel / grid column');ax.set_ylabel('pixel / grid row')
            fig.colorbar(im,ax=ax,label=unit,shrink=.8)
        rows=json.loads((args.run/'belief.json').read_text())
        ax=axes[1,1]
        if rows:
            start=rows[0]['t']
            for label,values in [('Expected source count',[sum(i*p for i,p in enumerate(r['cardinality']))
                                                           if r['cardinality'] else np.nan for r in rows]),
                                  ('MAP source count',[np.argmax(r['cardinality']) if r['cardinality'] else np.nan
                                                       for r in rows])]:
                ax.plot([r['t']-start for r in rows],values,label=label)
            ax.legend();ax.set_ylabel('sources');ax.set_xlabel('probe elapsed / s')
        else:ax.text(.5,.5,'Slow layer disabled',ha='center')
        ax.set_title('Slow posterior (observed sources)');ax.grid(alpha=.2)
        fig.suptitle(f"{args.run.name} | model {report['sensor_model'].upper()} | "
                     f"path {report['path_length_m']:.1f} m | software validation only")
        args.out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.out,dpi=150);plt.close(fig)
    print(args.out)


if __name__=='__main__':main()
