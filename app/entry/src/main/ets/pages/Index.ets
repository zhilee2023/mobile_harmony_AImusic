import native from 'libentry.so';
import { common } from '@kit.AbilityKit';
import { fileIo as fs, picker } from '@kit.CoreFileKit';
import { window } from '@kit.ArkUI';

interface VaeResult { success: boolean; error?: string; decoderOnlyComputeRtf?: number; sampledPeakRssKiB?: number; }
interface StageRtf { llm: number; dit: number; vae: number; other: number; total: number; }
interface ComparisonMemory { peakRssKiB: number; sampledPeakPssKiB?: number; sampledPeakGraphicsKiB?: number; }
interface ChainResult { success: boolean; error?: string; policy: string; totalSeconds?: number; sampledPeakRssKiB?: number; pipelineRtfThroughPcm?: number; validPcmSeconds?: number; strictNumericalParityPassed?: boolean; stageRtf?: StageRtf; memory?: ComparisonMemory; codecOnly?: boolean; codecOnlyRtf?: number; }
interface AppMemoryInfo { totalKiB: number; availableKiB: number; appRssLimitKiB: number; appVssLimitKiB: number; processPssKiB: number; processRssKiB: number; }
interface TestArchive { version: string; scope: string; high?: VaeResult; low?: VaeResult; chainSerial?: ChainResult; chainKeepLlm?: ChainResult; chainAll?: ChainResult; chainAllW4?: ChainResult; chunks5?: ChainResult; chunks5Tuned?: ChainResult; chunks5W8Vsin?: ChainResult; chunks5W8Snake?: ChainResult; llmShort?: ChainResult; llmLong?: ChainResult; comparisonLarge2?: ChainResult; comparisonSmall2?: ChainResult; comparisonLarge120?: ChainResult; comparisonSmall120?: ChainResult; comparisonLargeCodec120?: ChainResult; comparisonLargeCodec90?: ChainResult; comparisonLarge90?: ChainResult; comparisonSmall90?: ChainResult; memory?: AppMemoryInfo; memoryBeforeRun?: AppMemoryInfo; lastRunId?: string; }

@Entry
@Component
struct Index {
  @State lyrics: string = '[Verse]\n晚风轻轻吹过窗台\n星光落在你的眼海';
  @State style: number = 0;
  @State status: string = 'v0.1.11 · 30/90/120秒链路对照与退出原因记录；0.6B音乐模块尚未训练';
  @State memory: string = '';
  @State assets: string = '尚未检查测试文件';
  @State busy: boolean = false;
  @State comparisonSummary: string = '';
  @State comparisonSeconds: number = 30;
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
    try {
      const memoryFile = fs.openSync(`${this.context().filesDir}/startup-memory.json`, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
      try { fs.writeSync(memoryFile.fd, native.memoryProbe()); } finally { fs.closeSync(memoryFile); }
    } catch (_) {}
    const precision = AppStorage.get<string>('levoDiagnosticPrecision');
    if (precision === 'chain-comparison-large-2' || precision === 'chain-comparison-small-2' || precision === 'chain-comparison-large-2-120' || precision === 'chain-comparison-small-2-120' || precision === 'chain-comparison-large-2-120-low' || precision === 'chain-comparison-large-codec-120' || precision === 'chain-comparison-large-2-90' || precision === 'chain-comparison-small-2-90' || precision === 'chain-comparison-large-codec-90') {
      this.diagnosticRunId = AppStorage.get<string>('levoDiagnosticRunId') ?? '';
      AppStorage.delete('levoDiagnosticPrecision'); AppStorage.delete('levoDiagnosticRunId');
      this.runComparison(precision.substring(6)); return;
    }
    if (precision === 'high' || precision === 'low') {
      this.diagnosticRunId = AppStorage.get<string>('levoDiagnosticRunId') ?? '';
      AppStorage.delete('levoDiagnosticPrecision');
      AppStorage.delete('levoDiagnosticRunId');
      this.runVae(precision);
    } else if (precision === 'chain-serial' || precision === 'chain-keep-llm' || precision === 'chain-all' || precision === 'chain-all-w4' || precision === 'chain-llm-short' || precision === 'chain-llm-long' || precision === 'chain-chunks-5' || precision === 'chain-chunks-5-tuned' || precision === 'chain-chunks-5-w8-vsin' || precision === 'chain-chunks-5-w8-snake') {
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
      let archive: TestArchive = { version: '0.1.7', scope: 'VAE-only numerical stream; no DiT/LLM or speaker playback' };
      try { archive = JSON.parse(fs.readTextSync(reportPath)) as TestArchive; } catch (_) {}
      archive.version = '0.1.7';
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
      let archive: TestArchive = { version: '0.1.7', scope: 'Diagnostic module flow and memory; untrained music adapters, no quality validation' };
      try { archive = JSON.parse(fs.readTextSync(reportPath)) as TestArchive; } catch (_) {}
      archive.version = '0.1.7';
      archive.scope = 'Diagnostic module results; each case declares its scope; no trained music-quality proof';
      if (policy === 'serial') archive.chainSerial = result;
      else if (policy === 'keep-llm') archive.chainKeepLlm = result;
      else if (policy === 'all-w4') archive.chainAllW4 = result;
      else if (policy === 'llm-short') archive.llmShort = result;
      else if (policy === 'llm-long') archive.llmLong = result;
      else if (policy === 'chunks-5') archive.chunks5 = result;
      else if (policy === 'chunks-5-tuned') archive.chunks5Tuned = result;
      else if (policy === 'chunks-5-w8-vsin') archive.chunks5W8Vsin = result;
      else if (policy === 'chunks-5-w8-snake') archive.chunks5W8Snake = result;
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
  private async runComparison(variant: string): Promise<void> {
    if (this.busy) return; this.busy = true;
    const duration = variant.includes('-90') ? 90 : (variant.includes('-120') ? 120 : 30);
    this.comparisonSeconds = duration;
    this.status = variant.includes('small') ? `0.6B＋小DiT：${duration}秒等效数值链路，LM释放后运行两步DiT和VAE` : `公开预训练模型：${duration}秒歌曲链路，LM释放后运行两步DiT和VAE`;
    try {
      const win = await window.getLastWindow(this.context()); await win.setWindowKeepScreenOn(true);
      const raw = await native.runThreeModuleChain(`${this.context().filesDir}/chain-comparison`, variant);
      const result = JSON.parse(raw) as ChainResult;
      const path = `${this.context().filesDir}/levo-test-report.json`;
      let archive: TestArchive = { version: '0.1.8', scope: 'Matched serial chain comparison; small model is untrained and diagnostic only' };
      try { archive = JSON.parse(fs.readTextSync(path)) as TestArchive; } catch (_) {}
      archive.version = '0.1.11'; archive.lastRunId = this.diagnosticRunId;
      if (variant === 'comparison-large-codec-90') archive.comparisonLargeCodec90 = result;
      else if (variant === 'comparison-large-2-90') archive.comparisonLarge90 = result;
      else if (variant === 'comparison-small-2-90') archive.comparisonSmall90 = result;
      else if (variant === 'comparison-large-codec-120') archive.comparisonLargeCodec120 = result;
      else if (variant === 'comparison-small-2-120') archive.comparisonSmall120 = result;
      else if (variant === 'comparison-large-2-120' || variant === 'comparison-large-2-120-low') archive.comparisonLarge120 = result;
      else if (variant === 'comparison-small-2') archive.comparisonSmall2 = result; else archive.comparisonLarge2 = result;
      const file = fs.openSync(path, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
      try { fs.writeSync(file.fd, JSON.stringify(archive)); } finally { fs.closeSync(file); }
      if (result.success && result.codecOnly) {
        this.comparisonSummary = `独立DiT/VAE负载完成；Codec RTF ${result.codecOnlyRtf?.toFixed(3)}；不是完整生成RTF。`;
        this.status = '模块测试完成；是否使用重复夹具及输出路径见统一报告';
      } else if (result.success && result.stageRtf) {
        const rtf = result.stageRtf;
        const pss = result.memory?.sampledPeakPssKiB; const graphics = result.memory?.sampledPeakGraphicsKiB;
        const pssText = typeof pss === 'number' ? `${(pss / 1024).toFixed(1)} MiB` : '不可用';
        const graphicsText = typeof graphics === 'number' ? `${(graphics / 1024).toFixed(1)} MiB` : '不可用';
        this.comparisonSummary = `LLM RTF ${rtf.llm.toFixed(3)} · DiT ${rtf.dit.toFixed(3)} · VAE ${rtf.vae.toFixed(3)}\n总RTF ${rtf.total.toFixed(3)} · RSS峰值 ${((result.memory?.peakRssKiB ?? 0) / 1024).toFixed(1)} MiB\nPSS峰值 ${pssText} · 图形内存 ${graphicsText}`;
        this.status = '串行对照完成；详细配置与采样信息已写入统一报告';
      } else { this.status = `对照停止：${result.error ?? '查看报告'}`; }
    } catch (error) { this.status = `对照失败：${String(error)}`; }
    try { const win = await window.getLastWindow(this.context()); await win.setWindowKeepScreenOn(false); } catch (_) {}
    this.busy = false; this.diagnosticRunId = '';
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
        Text('串行链路对照').fontSize(22).fontWeight(FontWeight.Bold).fontColor('#F0F4FA').width('100%')
        Text('LM主体W4/头W8，DiT与VAE W8；单分支LM、两步DiT/CFG1.5。120秒固定3000码；小模型未训练，仅测流程。').fontSize(15).fontColor('#A9B9D2')
        Select([{ value: '30 秒对照' }, { value: '90 秒对照' }, { value: '120 秒对照' }]).selected([30, 90, 120].indexOf(this.comparisonSeconds))
          .enabled(!this.busy).onSelect((index: number) => { this.comparisonSeconds = [30, 90, 120][index]; })
        Text('公开模型沿用CPU High；CPU Low加载失败的补测入口已收起，开发复现仍保留。').fontSize(13).fontColor('#A9B9D2')
        Button('公开模型 · DiT两步').enabled(!this.busy).width('100%').onClick(() => { this.runComparison(this.comparisonSeconds === 90 ? 'comparison-large-2-90' : (this.comparisonSeconds === 120 ? 'comparison-large-2-120' : 'comparison-large-2')); })
        Button('0.6B＋小DiT · 两步').enabled(!this.busy).width('100%').onClick(() => { this.runComparison(this.comparisonSeconds === 90 ? 'comparison-small-2-90' : (this.comparisonSeconds === 120 ? 'comparison-small-2-120' : 'comparison-small-2')); })
        if (this.comparisonSummary.length > 0) { Text(this.comparisonSummary).fontSize(17).fontColor('#76E0C8').width('100%') }
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
        Text('推荐W8＋NEON原入口；较慢的Snake回调融合仅保留开发测试接口和统一报告。').fontColor('#A9B9D2').fontSize(13).width('100%')
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
