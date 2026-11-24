"""Scientific reference roofline; never compare mixed-precision graph times to FP32 ceilings."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.plot-deps').is_dir():sys.path.insert(0,str(ROOT/'.plot-deps'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-prefix',required=True);a=p.parse_args()
    r=json.loads(Path(a.input).read_text(encoding='utf-8'))['roofline'];compute=r['flops_per_second']/1e9;bandwidth=r['bytes_per_second']/1e9
    if any(not math.isfinite(v) or v<=0 for v in [compute,bandwidth]):raise ValueError('Positive finite calibration required')
    prefix=Path(a.output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    if any(prefix.with_suffix(ext).exists() for ext in ['.svg','.png']):raise ValueError('Preserve prior plotted measurements')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.logspace(-1,3,400);fig,ax=plt.subplots(figsize=(9,6))
    fig.subplots_adjust(top=.82,bottom=.19,left=.11,right=.97)
    ax.loglog(x,np.minimum(compute,bandwidth*x),color='#167D8D',lw=2.2,label='Attained FP32 / nominal-copy reference')
    ax.axhline(compute,ls=':',color='#6A7E87',alpha=.7)
    for point in r['points']:
        label=('0.6B' if point['model']=='small' else '4B')+' shape / '+point['phase']
        ax.scatter(point['arithmeticIntensity'],point['attainedGflops'],s=65,
                   marker='o' if point['model']=='small' else 's',color='#D47830' if point['phase']=='decode' else '#4459A6')
        offset=(8,12) if point['model']=='small' else (8,-18)
        ax.annotate(label,(point['arithmeticIntensity'],point['attainedGflops']),xytext=offset,textcoords='offset points',fontsize=9)
    ax.set(xlabel='Arithmetic intensity (FLOP / nominal byte)',ylabel='Attained FP32 GFLOP/s',
           title='MNN CPU High / 4 threads: isolated FP32 reference probes',xlim=(.1,1000),ylim=(1,400))
    ax.grid(True,which='both',alpha=.18);ax.legend(loc='upper left',frameon=False,fontsize=9)
    ax.title.set_fontsize(12)
    fig.suptitle(f'Compute reference {compute:.1f} GFLOP/s | nominal copy {bandwidth:.1f} GB/s',fontsize=11,y=.96)
    fig.text(.5,.025,'Real prefix values; synthetic weights. Not chip peaks or measured DRAM traffic.\nNot W4/W8 ceilings or a memory-bound proof.',ha='center',fontsize=8,color='#555555')
    fig.savefig(prefix.with_suffix('.svg'));fig.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(fig)
    svg=prefix.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print('Saved FP32 diagnostic reference plot; no quantized-LLM bottleneck classification')
if __name__=='__main__':main()
