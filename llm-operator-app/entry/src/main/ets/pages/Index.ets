import native from 'libentry.so';
import { common } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { fileIo as fs } from '@kit.CoreFileKit';
interface Ready { ready: boolean; error?: string; }
interface Round { prefillMs: number; decodeMsPerToken: number; decodeTokensPerSecond: number; }
interface Status { busy: boolean; status: string; phase?: string; round?: number; decodeDone?: number; decodeSteps?: number; rssPeakMiB?: number; roundResult?: Round; error?: string; }
@Entry
@Component
struct Index {
  @State model: number = 0;
  @State precision: string = 'high';
  @State threads: number = 4;
  @State steps: number = 64;
  @State ready: boolean = false;
  @State note: string = '正在检查基线模型';
  @State busy: boolean = false;
  @State status: string = '准备实验';
  @State done: number = 0;
  @State total: number = 64;
  @State round: number = 0;
  @State prefill: number = 0;
  @State decode: number = 0;
  @State rate: number = 0;
  @State ram: number = 0;
  private timer: number = -1;
  private root(): string { return (getContext(this) as common.UIAbilityContext).filesDir; }
  private refresh(): void { const r = JSON.parse(native.inspect(this.root(), this.model)) as Ready; this.ready = r.ready; this.note = r.ready ? '基线模型与固定输入已就绪' : r.error ?? '等待USB部署基线文件'; }
  private async awake(enable: boolean): Promise<void> { try { const w = await window.getLastWindow(getContext(this) as common.UIAbilityContext); await w.setWindowKeepScreenOn(enable); } catch (_) {} }
  aboutToAppear(): void {
    this.refresh(); this.timer = setInterval(() => { this.poll(); }, 500);
    const mode = AppStorage.get<string>('operatorLabMode');
    if (mode === 'baseline' || mode === 'operators') {
      AppStorage.delete('operatorLabMode'); this.model = AppStorage.get<number>('operatorLabModel') ?? 0;
      this.precision = AppStorage.get<string>('operatorLabPrecision') ?? 'high'; this.threads = AppStorage.get<number>('operatorLabThreads') ?? 4;
      this.steps = AppStorage.get<number>('operatorLabSteps') ?? 64; this.refresh();
      setTimeout(() => { this.start(mode, AppStorage.get<number>('operatorLabRounds') ?? 3); }, 600);
    }
  }
  aboutToDisappear(): void { if (this.timer >= 0) clearInterval(this.timer); native.cancel(); this.awake(false); }
  private poll(): void {
    try {
      const s = JSON.parse(native.poll()) as Status; this.busy = s.busy; this.status = s.error ?? `${s.status} · ${s.phase ?? ''}`;
      this.done = s.decodeDone ?? 0; this.total = s.decodeSteps ?? this.steps; this.round = s.round ?? 0; this.ram = s.rssPeakMiB ?? 0;
      if (s.roundResult) { this.prefill = s.roundResult.prefillMs; this.decode = s.roundResult.decodeMsPerToken; this.rate = s.roundResult.decodeTokensPerSecond; }
      const p = `${this.root()}/last-state.json`; const pending = `${p}.pending`; const f = fs.openSync(pending, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
      try { fs.writeSync(f.fd, JSON.stringify(s)); } finally { fs.closeSync(f); }
      fs.renameSync(pending, p);
    } catch (_) {}
  }
  private async start(mode: string, rounds: number): Promise<void> {
    if (this.busy || !this.ready) return; this.busy = true; await this.awake(true);
    try {
      await native.start(this.root(), this.model, JSON.stringify({ mode: mode, decodeSteps: mode === 'operators' ? 2 : this.steps, rounds: mode === 'operators' ? 1 : rounds,
        precision: this.precision, threads: this.threads, runTag: AppStorage.get<string>('operatorLabRunTag') ?? '', maxRssMiB: 6144, minAvailableMiB: 650 }));
    } catch (error) { this.status = String(error); }
    finally { this.busy = false; this.poll(); await this.awake(false); }
  }
  build() {
    Scroll() {
      Column({ space: 18 }) {
        Text('LLM 算子实验').fontSize(29).fontWeight(FontWeight.Bold).width('100%')
        Text('独立App · B0 MNN baseline').fontSize(15).fontColor('#5E7374').width('100%')
        Select([{ value: '0.6B · 28层音乐Qwen' }, { value: '4B · 36层音乐Qwen' }]).selected(this.model).width('100%').enabled(!this.busy).onSelect((index: number) => { this.model = index; this.refresh(); })
        Text(this.note).fontSize(13).width('100%').onClick(() => { this.refresh(); })
        Row({ space: 12 }) {
          Select([{ value: 'CPU High' }, { value: 'CPU Low' }]).selected(this.precision === 'high' ? 0 : 1).enabled(!this.busy).onSelect((index: number) => { this.precision = index === 0 ? 'high' : 'low'; }).layoutWeight(1)
          Select([{ value: '2线程' }, { value: '4线程' }, { value: '6线程' }]).selected([2,4,6].indexOf(this.threads)).enabled(!this.busy).onSelect((index: number) => { this.threads = [2,4,6][index]; }).width(100)
        }.width('100%')
        Select([{ value: '64次decode' }, { value: '128次decode' }, { value: '256次decode' }]).selected([64,128,256].indexOf(this.steps)).enabled(!this.busy).onSelect((index: number) => { this.steps = [64,128,256][index]; }).width('100%')
        Button('运行基线 B0').enabled(this.ready && !this.busy).width('100%').onClick(() => { this.start('baseline', 3); })
        Button('算子诊断（单独计时）').enabled(this.ready && !this.busy).width('100%').onClick(() => { this.start('operators', 1); })
        Button('停止').enabled(this.busy).width('100%').onClick(() => { native.cancel(); this.status = '已请求停止，等待当前前向返回'; })
        Text(this.status).fontSize(18).width('100%')
        Progress({ value: this.done, total: Math.max(1,this.total), type: ProgressType.Linear }).width('100%')
        Text(`第${this.round}轮 · decode ${this.done}/${this.total}`).fontSize(14).width('100%')
        Text(`Prefill ${this.prefill.toFixed(2)} ms`).fontSize(18).width('100%')
        Text(`纯Decode ${this.decode.toFixed(2)} ms/token · ${this.rate.toFixed(2)} token/s`).fontSize(16).width('100%')
        Text(`进程RSS峰值 ${this.ram.toFixed(0)} MiB`).fontSize(16).width('100%')
        Text('固定真实token历史；不生成音乐。首次与预热分开，算子诊断不混入基线成绩。').fontSize(12).fontColor('#5E7374').width('100%')
        Text('统一导出：llm-operator-report.json。电脑端USB脚本自动取回；CG、DP及候选库搜索在电脑执行。').fontSize(12).fontColor('#5E7374').width('100%')
      }.padding(24).width('100%')
    }.width('100%').height('100%').backgroundColor('#F2F6F5')
  }
}
