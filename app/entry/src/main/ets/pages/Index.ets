import native from 'libentry.so';
import { common } from '@kit.AbilityKit';
import { fileIo as fs, picker } from '@kit.CoreFileKit';

interface VaeResult { success: boolean; error?: string; decoderOnlyComputeRtf?: number; sampledPeakRssKiB?: number; }
interface ChainResult { success: boolean; error?: string; policy: string; totalSeconds?: number; sampledPeakRssKiB?: number; pipelineRtfThroughPcm?: number; validPcmSeconds?: number; strictNumericalParityPassed?: boolean; }
interface AppMemoryInfo { totalKiB: number; availableKiB: number; appRssLimitKiB: number; appVssLimitKiB: number; processPssKiB: number; processRssKiB: number; }
interface TestArchive { version: string; scope: string; high?: VaeResult; low?: VaeResult; chainSerial?: ChainResult; chainKeepLlm?: ChainResult; chainAll?: ChainResult; chainAllW4?: ChainResult; chunks5?: ChainResult; chunks5Tuned?: ChainResult; chunks5W8Vsin?: ChainResult; llmShort?: ChainResult; llmLong?: ChainResult; memory?: AppMemoryInfo; memoryBeforeRun?: AppMemoryInfo; lastRunId?: string; }

@Entry
@Component
struct Index {
  @State lyrics: string = '[Verse]\n晚风轻轻吹过窗台\n星光落在你的眼海';
  @State style: number = 0;
  @State status: string = 'v0.1.6 · 数值与内存诊断；尚无训练好的0.6B音乐权重';
  @State memory: string = '';
  @State assets: string = '尚未检查测试文件';
  @State busy: boolean = false;
  private diagnosticRunId: string = '';
  private required: string[] = ['fixture.json', 'vae-256-fp32.mnn', 'vae-256-fp32.mnn.weight',
    'latent-0.f32', 'latent-1.f32', 'latent-2.f32', 'pcm-0.f32', 'pcm-1.f32', 'pcm-2.f32',
    'emit-0.f32', 'emit-1.f32', 'emit-2.f32'];
  private context(): common.UIAbilityContext { return getContext(this) as common.UIAbilityContext; }
  private modelDir(): string { return `${this.context().filesDir}/vae-stream-256`; }
  private refresh(): void {
    let count: number = 0;
    this.required.forEach((name: string) => { try { if (fs.accessSync(`${this.modelDir()}/${name}`)) count++; } catch (_) {} });
    this.assets = `测试文件 ${count}/${this.required.length} · VAE 256帧／10.24秒窗`;
  }
  aboutToAppear(): void {
    this.refresh();
    const precision = AppStorage.get<string>('levoDiagnosticPrecision');
    if (precision === 'high' || precision === 'low') {
      this.diagnosticRunId = AppStorage.get<string>('levoDiagnosticRunId') ?? '';
      AppStorage.delete('levoDiagnosticPrecision');
      AppStorage.delete('levoDiagnosticRunId');
      this.runVae(precision);
    } else if (precision === 'chain-serial' || precision === 'chain-keep-llm' || precision === 'chain-all' || precision === 'chain-all-w4' || precision === 'chain-llm-short' || precision === 'chain-llm-long' || precision === 'chain-chunks-5' || precision === 'chain-chunks-5-tuned' || precision === 'chain-chunks-5-w8-vsin') {
      this.diagnosticRunId = AppStorage.get<string>('levoDiagnosticRunId') ?? '';
      AppStorage.delete('levoDiagnosticPrecision');
      AppStorage.delete('levoDiagnosticRunId');
      this.runChain(precision.substring(6));
    }
  }
  private async importFiles(): Promise<void> {
    this.busy = true;
    try {
      const view = new picker.DocumentViewPicker(this.context());
      const uris = await view.select({ maxSelectNumber: 20 });
      try { fs.mkdirSync(this.modelDir()); } catch (_) {}
      let imported: number = 0;
      uris.forEach((uri: string) => {
        const name = decodeURIComponent(uri.substring(uri.lastIndexOf('/') + 1));
        if (this.required.includes(name)) { fs.copyFileSync(uri, `${this.modelDir()}/${name}`); imported++; }
      });
      this.status = `已复制 ${imported} 个匹配文件；识别数量不代表数值验收`;
      this.refresh();
    } catch (error) { this.status = `导入失败：${String(error)}`; }
    this.busy = false;
  }
  private async runVae(precision: string): Promise<void> {
    this.busy = true;
    this.status = `正在后台运行 VAE ${precision}：3个窗，保留64帧尾部；不播放数值夹具`;
    try {
      const memoryBeforeRun = JSON.parse(native.memoryProbe()) as AppMemoryInfo;
      const raw: string = await native.runVaeStream(this.modelDir(), precision);
      const report = JSON.parse(raw) as VaeResult;
      const reportPath = `${this.context().filesDir}/levo-test-report.json`;
      let archive: TestArchive = { version: '0.1.6', scope: 'VAE-only numerical stream; no DiT/LLM or speaker playback' };
      try { archive = JSON.parse(fs.readTextSync(reportPath)) as TestArchive; } catch (_) {}
      archive.version = '0.1.6';
      archive.scope = 'Diagnostic module results; each case declares its scope; no trained music-quality proof';
      archive.memoryBeforeRun = memoryBeforeRun;
      archive.lastRunId = this.diagnosticRunId;
      if (precision === 'high') archive.high = report; else archive.low = report;
      archive.memory = JSON.parse(native.memoryProbe()) as AppMemoryInfo;
      const file = fs.openSync(reportPath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
      try { fs.writeSync(file.fd, JSON.stringify(archive)); } finally { fs.closeSync(file); }
      if (report.success) {
        this.status = `VAE数值分块完成：有效25.60秒，模块计算RTF ${report.decoderOnlyComputeRtf?.toFixed(3)}，采样峰值RSS ${((report.sampledPeakRssKiB ?? 0) / 1024).toFixed(1)} MiB。未包含DiT/LLM和扬声器播放。`;
      } else { this.status = `测试未通过：${report.error ?? '查看统一报告'}`; }
    } catch (error) { this.status = `测试失败：${String(error)}`; }
    this.memory = native.memoryProbe();
    this.diagnosticRunId = '';
    this.busy = false;
  }
  private async exportReport(): Promise<void> {
    try {
      const view = new picker.DocumentViewPicker(this.context());
      const uris = await view.save({ newFileNames: ['levo-test-report.json'] });
      if (uris.length) fs.copyFileSync(`${this.context().filesDir}/levo-test-report.json`, uris[0]);
    } catch (error) { this.status = `导出失败：${String(error)}`; }
  }
  private async runChain(policy: string): Promise<void> {
    this.busy = true;
    this.status = policy.startsWith('chunks-5') ? `连续5块 ${policy}：一次prefill，LLM 1024码，256帧窗口/64帧重叠→40.96秒数值PCM；音乐头未训练` : `三模块诊断 ${policy}：512歌词占位token＋风格＋BOS，LLM 256码→完整DiT两步→VAE；音乐头未训练`;
    try {
      const memoryBeforeRun = JSON.parse(native.memoryProbe()) as AppMemoryInfo;
      const raw: string = await native.runThreeModuleChain(`${this.context().filesDir}/three-module`, policy);
      const result = JSON.parse(raw) as ChainResult;
      const reportPath = `${this.context().filesDir}/levo-test-report.json`;
      let archive: TestArchive = { version: '0.1.6', scope: 'Diagnostic module flow and memory; untrained music adapters, no quality validation' };
      try { archive = JSON.parse(fs.readTextSync(reportPath)) as TestArchive; } catch (_) {}
      archive.version = '0.1.6';
      archive.scope = 'Diagnostic module results; each case declares its scope; no trained music-quality proof';
      if (policy === 'serial') archive.chainSerial = result;
      else if (policy === 'keep-llm') archive.chainKeepLlm = result;
      else if (policy === 'all-w4') archive.chainAllW4 = result;
      else if (policy === 'llm-short') archive.llmShort = result;
      else if (policy === 'llm-long') archive.llmLong = result;
      else if (policy === 'chunks-5') archive.chunks5 = result;
      else if (policy === 'chunks-5-tuned') archive.chunks5Tuned = result;
      else if (policy === 'chunks-5-w8-vsin') archive.chunks5W8Vsin = result;
      else archive.chainAll = result;
      archive.memoryBeforeRun = memoryBeforeRun;
      archive.memory = JSON.parse(native.memoryProbe()) as AppMemoryInfo;
      archive.lastRunId = this.diagnosticRunId;
      const file = fs.openSync(reportPath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
      try { fs.writeSync(file.fd, JSON.stringify(archive)); } finally { fs.closeSync(file); }
      this.status = result.success ? `流程/内存完成 ${policy}；峰值RSS ${((result.sampledPeakRssKiB ?? 0) / 1024).toFixed(1)} MiB，整链RTF ${result.pipelineRtfThroughPcm?.toFixed(3) ?? '不适用'}，有效PCM ${result.validPcmSeconds?.toFixed(2) ?? '未生成'}秒。严格数值对齐：${result.strictNumericalParityPassed ? '通过' : '未通过或不适用'}；不是音乐质量验收。` : `三模块安全停止：${result.error}`;
    } catch (error) { this.status = `三模块测试失败：${String(error)}`; }
    this.memory = native.memoryProbe();
    this.diagnosticRunId = '';
    this.busy = false;
  }
  build() {
    Scroll() {
      Column({ space: 18 }) {
        Text('LeVo Harmony').fontSize(30).fontWeight(FontWeight.Bold).fontColor('#F0F4FA')
        Text('混合歌曲 · 单路音乐Token · 0.6B骨干研究').fontColor('#A9B9D2')
        TextArea({ text: this.lyrics, placeholder: '歌词，最多512个分词token' })
          .height(190).onChange((text: string) => { this.lyrics = text; })
        Select([{ value: '中文流行' }, { value: '民谣' }, { value: '摇滚' }, { value: '电子' }])
          .selected(this.style).onSelect((index: number) => { this.style = index; })
        Text('一个风格ID对应一个可训练前缀向量；当前标签尚未完成音乐训练映射。').fontColor('#A9B9D2').fontSize(13)
        Button('读取设备内存').width('100%').height(48)
          .onClick(() => { this.memory = native.memoryProbe(); })
        Text(this.memory).fontColor('#76E0C8').fontSize(13).width('100%')
        Text(this.status).fontColor('#E9C882').width('100%')
        Text('10.24 秒分块实验').fontSize(22).fontWeight(FontWeight.Bold).fontColor('#F0F4FA').width('100%')
        Text(this.assets).fontColor('#A9B9D2').width('100%')
        Button('导入12个测试文件').enabled(!this.busy).width('100%').onClick(() => { this.importFiles(); })
        Button('刷新文件状态').enabled(!this.busy).width('100%').onClick(() => { this.refresh(); })
        Button('VAE CPU High · 三窗数值与内存').enabled(!this.busy).width('100%').onClick(() => { this.runVae('high'); })
        Button('VAE CPU Low · 三窗数值与内存').enabled(!this.busy).width('100%').onClick(() => { this.runVae('low'); })
        Text('三模块内存诊断 · 需预装独立测试包').fontColor('#A9B9D2')
        Button('三模块 · 每阶段释放').enabled(!this.busy).width('100%').onClick(() => { this.runChain('serial'); })
        Button('三模块 · 保留LLM和KV').enabled(!this.busy).width('100%').onClick(() => { this.runChain('keep-llm'); })
        Button('三模块 · 全部共驻').enabled(!this.busy).width('100%').onClick(() => { this.runChain('all'); })
        Button('三模块 · DiT W4全部共驻').enabled(!this.busy).width('100%').onClick(() => { this.runChain('all-w4'); })
        Button('连续5块 · 同一KV与三模型共驻').enabled(!this.busy).width('100%').onClick(() => { this.runChain('chunks-5'); })
        Button('连续5块 · VAE W8＋NEON正弦').enabled(!this.busy).width('100%').onClick(() => { this.runChain('chunks-5-w8-vsin'); })
        Button('导出统一测试报告').enabled(!this.busy).width('100%').onClick(() => { this.exportReport(); })
        Text('当前实验使用真实VAE和合成latent；先测数值与有界内存，不是完整DiT/LLM生成，也未验证播放器。')
          .fontColor('#A9B9D2').fontSize(13).width('100%')
        Text('歌词 → 1个风格前缀＋Qwen3骨干 → 混合音乐Token → DiT → 48kHz双声道VAE')
          .fontColor('#CCD7E8').fontSize(15)
        Text('Qwen文本预训练权重可加载；音乐embedding、输出头和风格向量必须训练。此版本不显示虚假的生成/播放结果。')
          .fontColor('#A9B9D2').fontSize(13)
      }.padding(24).width('100%')
    }.width('100%').height('100%').backgroundColor('#101A2C')
  }
}
