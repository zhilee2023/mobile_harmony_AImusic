import native from 'libentry.so';
import { common } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { media } from '@kit.MediaKit';
import { fileIo as fs, picker } from '@kit.CoreFileKit';
import { StreamPlayer, StreamStatus, PlaybackStatus } from './StreamPlayer';
import { SHORT_LYRICS, LONG_LYRICS } from './LyricPresets';

interface ModelStatus { ready: boolean; count: number; total: number; error?: string; }
interface StyleChoice { name: string; path: string; }
interface StreamEvent { kind: string; timeMs: number; runId: string; busy: boolean; generatedMs: number; playedMs: number; queueMs: number; detail: string; }
@Entry
@Component
struct Index {
  @State model: number = 0;
  @State seconds: number = 10.2;
  @State lyrics: string = SHORT_LYRICS;
  @State lyricPreset: number = 0;
  @State fillDuration: boolean = true;
  @State styles: StyleChoice[] = [];
  @State styleIndex: number = 0;
  @State modelReady: boolean = false;
  @State modelNote: string = '正在检查模型';
  @State busy: boolean = false;
  @State stopping: boolean = false;
  @State stage: string = '准备实验';
  @State generatedMs: number = 0;
  @State playedMs: number = 0;
  @State conditionFrames: number = 0;
  @State playing: boolean = false;
  @State waiting: boolean = false;
  @State elapsed: number = 0;
  @State firstChunk: number = 0;
  @State rss: number = 0;
  @State lm: number = 0;
  @State dit: number = 0;
  @State vae: number = 0;
  @State queueMs: number = 0;
  @State playbackError: string = '';
  private player: StreamPlayer | undefined = undefined;
  private systemPlayer: media.AVPlayer | undefined = undefined;
  private systemFile: fs.File | undefined = undefined;
  private timer: number = -1;
  private testStop: boolean = false;
  private seekDone: boolean = false;
  private runId: string = '';
  private lastLog: number = 0;
  private checkpointAt: number = 0;
  private checkpointRun: string = '';
  private checkpointStatus: string = '';
  private checkpointBusy: boolean = false;
  private awakeRelease: number = -1;
  private durationBenchmark: boolean = false;
  private lmDynamicOption: number = 2;
  private vaeFusion: boolean = true;
  private benchmarkRunTag: string = '';
  private startedAt: number = 0;
  @State dragging: boolean = false;
  @State previewMs: number = 0;
  private root(): string { return (getContext(this) as common.UIAbilityContext).filesDir; }
  private croppedLm(): string {
    for (const name of ['music-head-audio-r2', 'music-head-r1']) {
      const folder = `${this.root()}/experimental/${name}`;
      try { if (fs.accessSync(`${folder}/head-range.json`)) return folder; } catch (_) {}
    }
    return '';
  }
  private time(ms: number): string { return `${Math.floor(ms / 60000)}:${Math.floor(ms / 1000 % 60).toString().padStart(2, '0')}`; }
  private durations(): number[] { return this.model === 0 ? [10.2, 15, 30, 60, 90, 120, 240, 300] : [10.2, 15, 30, 60, 90, 120]; }
  private phaseName(phase: string): string {
    switch (phase) {
      case 'loading': return '加载模型'; case 'prefill': return '前缀计算'; case 'decode': return '生成条件帧';
      case 'projection': return '条件投影'; case 'dit': return '分块扩散'; case 'vae': return '音频解码'; case 'published': return '片段可播放';
      default: return '正在生成';
    }
  }
  private record(kind: string, detail: string = ''): void {
    try {
      const item: StreamEvent = { kind: kind, timeMs: Date.now(), runId: this.runId, busy: this.busy,
        generatedMs: this.generatedMs, playedMs: this.playedMs, queueMs: this.queueMs, detail: detail };
      const file = fs.openSync(`${this.root()}/stream-events.jsonl`, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.APPEND);
      try { fs.writeSync(file.fd, JSON.stringify(item) + '\n'); } finally { fs.closeSync(file); }
    } catch (_) {}
  }
  private refresh(): void {
    const result = JSON.parse(native.inspect(this.root(), this.model)) as ModelStatus;
    this.modelReady = result.ready;
    this.modelNote = result.ready ? '同规格模型文件已就绪' : `模型文件 ${result.count ?? 0}/${result.total ?? 0} · 等待部署`;
    this.styles = JSON.parse(native.styles(this.root())) as StyleChoice[];
    if (this.styleIndex >= this.styles.length) this.styleIndex = 0;
    if (this.styleIndex === 0) {
      for (let i = 0; i < this.styles.length; i++) if (this.styles[i].name === 'tanhai') this.styleIndex = i;
      if (this.model === 0) for (let i = 0; i < this.styles.length; i++) if (this.styles[i].name === 'muq-text-warm-pop') this.styleIndex = i;
    }
  }
  private async importStyle(): Promise<void> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const choices = await new picker.DocumentViewPicker(context).select({ maxSelectNumber: 1, fileSuffixFilters: ['NPY|.npy'] });
      if (choices.length === 0) return;
      const source = fs.openSync(choices[0], fs.OpenMode.READ_ONLY);
      try {
        if (fs.statSync(source.fd).size > 65536) throw new Error('需要512维float32 NPY文件');
        const folder = `${this.root()}/clap-styles`; if (!fs.accessSync(folder)) fs.mkdirSync(folder, true);
        const name = `imported-${Date.now()}`; fs.copyFileSync(source.fd, `${folder}/${name}.npy`); this.refresh();
        for (let i = 0; i < this.styles.length; i++) if (this.styles[i].name === name) this.styleIndex = i;
      } finally { fs.closeSync(source); }
    } catch (error) { this.stage = String(error); }
  }
  aboutToAppear(): void {
    this.refresh(); native.restore(this.root()); this.poll(); this.timer = setInterval(() => { this.poll(); }, 100);
    const autoRun = AppStorage.get<string>('streamLabAutoRun');
    if (autoRun === 'replay-last') { AppStorage.delete('streamLabAutoRun'); setTimeout(() => { this.togglePlayback(); }, 500); }
    else if (autoRun === 'system-last') { AppStorage.delete('streamLabAutoRun'); setTimeout(() => { this.systemListen(); }, 500); }
    else if (autoRun) {
      AppStorage.delete('streamLabAutoRun');
      this.model = AppStorage.get<number>('streamLabModel') ?? 0;
      this.durationBenchmark = autoRun === 'duration-benchmark';
      this.seconds = this.durationBenchmark ? (AppStorage.get<number>('streamLabSeconds') ?? 30) : autoRun === 'mnn60-check' ? 60 : autoRun === 'music-check' ? 10.2 : 15;
      if (this.durationBenchmark && [30, 60, 90, 120, 240, 300].indexOf(this.seconds) < 0) { this.stage = '不支持的测速时长'; return; }
      if (this.durationBenchmark && (this.seconds >= 240 || (this.model === 1 && this.seconds >= 120))) { this.lyrics = LONG_LYRICS; this.lyricPreset = 1; }
      this.lmDynamicOption = AppStorage.get<number>('streamLabDynamicOption') ?? 2;
      this.vaeFusion = AppStorage.get<boolean>('streamLabVaeFusion') ?? true;
      this.benchmarkRunTag = this.durationBenchmark ? (AppStorage.get<string>('streamLabRunTag') ?? '') : '';
      this.testStop = autoRun === 'stop-check'; this.refresh();
      setTimeout(() => { this.start(!this.durationBenchmark && autoRun !== 'mnn-check' && autoRun !== 'mnn60-check' && autoRun !== 'music-check'); }, 800);
    }
  }
  aboutToDisappear(): void {
    if (this.timer >= 0) clearInterval(this.timer);
    if (this.awakeRelease >= 0) clearTimeout(this.awakeRelease);
    native.cancel(); const old = this.player; this.player = undefined; old?.close(); this.releaseSystem();
  }
  private poll(): void {
    try {
      const state = JSON.parse(native.poll()) as StreamStatus;
      if (!state.runId) return;
      this.runId = state.runId;
      this.busy = state.busy; this.generatedMs = state.generatedSamples / 48;
      this.conditionFrames = state.conditionFrames ?? 0;
      this.elapsed = state.busy && this.startedAt > 0 ? (Date.now() - this.startedAt) / 1000 : state.elapsedSeconds ?? 0;
      this.firstChunk = state.firstChunkSeconds ?? 0; this.rss = state.rssPeakMiB ?? 0;
      this.lm = state.lmSeconds ?? 0; this.dit = state.ditSeconds ?? 0; this.vae = state.vaeSeconds ?? 0;
      this.stage = state.busy ? (this.stopping ? '正在停止，保留已生成片段…' : this.phaseName(state.phase ?? '')) :
        state.status === 'completed' ? '本轮完成' : state.status === 'stopped' ? '已停止，可回放现有片段' : `实验停止：${state.error ?? ''}`;
      this.player?.update(state);
      if (Date.now() - this.checkpointAt > 1000 && (state.busy || this.checkpointRun !== state.runId || this.checkpointStatus !== state.status || this.checkpointBusy !== state.busy)) {
        try {
          const pending = `${this.root()}/last-state.json.pending`;
          const file = fs.openSync(pending, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
          try { fs.writeSync(file.fd, JSON.stringify(state)); } finally { fs.closeSync(file); }
          fs.renameSync(pending, `${this.root()}/last-state.json`);
          this.checkpointAt = Date.now(); this.checkpointRun = state.runId; this.checkpointStatus = state.status; this.checkpointBusy = state.busy;
        } catch (_) {}
      }
      if (this.testStop && state.generatedSamples > 0 && state.busy) { this.testStop = false; this.stop(); }
      if (!this.seekDone && AppStorage.get<boolean>('streamLabSeekCheck') && state.generatedSamples > 48000) {
        this.seekDone = true;
        const future = this.generatedMs + 5000; this.player?.seek(future); this.record('seek-future-request', `${future}`);
        setTimeout(() => { this.player?.seek(500); this.record('seek-back-request', '500'); }, 300);
      }
    } catch (_) {}
  }
  private async keepAwake(enable: boolean): Promise<void> {
    try { const context = getContext(this) as common.UIAbilityContext; const w = await window.getLastWindow(context); await w.setWindowKeepScreenOn(enable); } catch (_) {}
  }
  private async releaseSystem(): Promise<void> {
    if (this.systemPlayer) { try { await this.systemPlayer.release(); } catch (_) {} this.systemPlayer = undefined; }
    if (this.systemFile) { fs.closeSync(this.systemFile); this.systemFile = undefined; }
  }
  private async togglePlayback(): Promise<void> {
    try {
      await this.releaseSystem();
      if (!this.player) {
        this.player = this.createPlayer(); this.player.update(JSON.parse(native.poll()) as StreamStatus); await this.player.open(); this.player.play();
      } else if (this.playing) this.player.pause(); else this.player.play();
    } catch (error) { this.playbackError = String(error); this.record('playback-error', String(error)); }
  }
  private createPlayer(): StreamPlayer {
    return new StreamPlayer((p: PlaybackStatus): void => {
      this.playedMs = p.sample / 48; this.playing = p.playing; this.waiting = p.waiting; this.queueMs = p.submittedAheadSamples / 48;
      if (p.error.length > 0) { this.stage = p.error; this.playbackError = p.error; }
      if (Date.now() - this.lastLog >= 300 || p.error.length > 0) { this.lastLog = Date.now(); this.record(p.error.length > 0 ? 'playback-error' : 'playback', p.error || ((p.recoveries ?? 0) > 0 ? `rendererRecoveries=${p.recoveries}` : '')); }
    });
  }
  private async systemListen(): Promise<void> {
    try {
      if (this.player) { await this.player.close(); this.player = undefined; }
      await this.releaseSystem(); const status = JSON.parse(native.poll()) as StreamStatus;
      if (status.chunks.length === 0) return;
      const chunk = status.chunks[0]; const player = await media.createAVPlayer(); this.systemPlayer = player;
      player.on('stateChange', async (state: string) => {
        this.record('system-player', state);
        try { if (state === 'initialized') await player.prepare(); if (state === 'prepared') { await player.setVolume(1); await player.play(); } }
        catch (error) { this.playbackError = String(error); }
      });
      player.on('timeUpdate', (position: number) => { this.playedMs = position; this.record('system-position'); });
      player.on('error', (error) => { this.playbackError = error.message; this.record('system-error', error.message); });
      this.systemFile = fs.openSync(chunk.path, fs.OpenMode.READ_ONLY);
      player.fdSrc = { fd: this.systemFile.fd, offset: 0, length: fs.statSync(chunk.path).size };
    } catch (error) { this.playbackError = String(error); }
  }
  private async start(transport: boolean): Promise<void> {
    if (this.busy || (!transport && !this.modelReady)) return;
    const fixedBenchmark = this.durationBenchmark;
    if (this.awakeRelease >= 0) { clearTimeout(this.awakeRelease); this.awakeRelease = -1; }
    this.busy = true; this.stopping = false; this.generatedMs = 0; this.playedMs = 0; this.seekDone = false;
    this.startedAt = Date.now(); this.playbackError = '';
    this.firstChunk = 0; this.stage = '准备实验'; await this.keepAwake(true);
    if (this.player) await this.player.close(); await this.releaseSystem(); this.player = this.createPlayer();
    try { await this.player.open(); await native.start(this.root(), this.model, Math.round(this.seconds * 25), transport, this.lyrics, this.styles[this.styleIndex]?.path ?? '', JSON.stringify({ resident: this.model === 0, ditSteps: 2, cfg: true, croppedLm: this.model === 0 ? this.croppedLm() : '', lmDynamicOption: this.model === 0 ? this.lmDynamicOption : 0, vaeFusion: this.model === 0 && this.vaeFusion, longRun: this.seconds > 60, suppressEos: fixedBenchmark || this.fillDuration, runTag: fixedBenchmark ? this.benchmarkRunTag : '' })); }
    catch (error) { this.stage = String(error); this.playbackError = String(error); this.record('start-error', String(error)); }
    finally { this.busy = false; this.stopping = false; this.durationBenchmark = false; this.poll(); if (fixedBenchmark) this.awakeRelease = setTimeout(() => { this.awakeRelease = -1; this.keepAwake(false); }, 15000); else await this.keepAwake(false); }
  }
  private stop(): void { this.stopping = true; native.cancel(); this.record('stop-request'); }
  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column({ space: 8 }) {
          Text($r('app.string.app_name')).fontSize(29).fontWeight(FontWeight.Bold).fontColor('#172C30')
          Text('歌词与风格驱动 · 分段生成与回放').fontSize(14).fontColor('#667D80')
        }.width('100%').alignItems(HorizontalAlign.Start)
        Row({ space: 12 }) {
          Select([{ value: '0.6B · 28层 + 22层 DiT' }, { value: '4B · 36层 + 39层 DiT' }])
            .selected(this.model).value(this.model === 0 ? '0.6B · 28/22层' : '4B · 36/39层').layoutWeight(1).enabled(!this.busy).onSelect((index: number) => { this.model = index; if (this.model !== 0 && this.seconds > 120) this.seconds = 120; this.styleIndex = 0; this.refresh(); })
          Select(this.durations().map((seconds: number): SelectOption => { return { value: `${seconds}秒` }; })).selected(this.durations().indexOf(this.seconds))
            .width(82).enabled(!this.busy).onSelect((index: number) => { this.seconds = this.durations()[index]; })
        }.width('100%')
        Text(this.modelNote + ' · 点此刷新').fontSize(13).fontColor('#667D80').width('100%').onClick(() => { this.refresh(); })
        Column({ space: 10 }) {
          Text('歌词').fontSize(17).width('100%')
          Select([{ value: '短歌词 · 原对照' }, { value: '长歌词 · 完整段落' }, { value: '自定义歌词' }]).selected(this.lyricPreset).width('100%').enabled(!this.busy)
            .onSelect((index: number) => { this.lyricPreset = index; if (index < 2) this.lyrics = index === 0 ? SHORT_LYRICS : LONG_LYRICS; })
          TextArea({ text: this.lyrics, placeholder: '输入歌词，支持[verse]、[chorus]等段落标签' }).height(180).maxLength(1800).enabled(!this.busy).onChange((value: string) => { if (value !== this.lyrics) this.lyricPreset = 2; this.lyrics = value; })
          Row({ space: 10 }) {
            Select(this.styles.map((item: StyleChoice): SelectOption => { return { value: item.name === 'muq-text-warm-pop' ? 'MuQ文本·温暖流行' : item.name }; })).selected(this.styleIndex).value(this.styles[this.styleIndex]?.name === 'muq-text-warm-pop' ? 'MuQ文本·温暖流行' : this.styles[this.styleIndex]?.name ?? '模型默认')
              .layoutWeight(1).enabled(!this.busy).onSelect((index: number) => { this.styleIndex = index; })
            Button('导入NPY').enabled(!this.busy).onClick(() => { this.importStyle(); })
          }.width('100%')
          Text('已带入原LarkStudio测试歌词和风格文件；可修改后再生成。').fontSize(12).fontColor('#667D80')
          Row({ space: 10 }) {
            Toggle({ type: ToggleType.Switch, isOn: this.fillDuration }).enabled(!this.busy).onChange((value: boolean) => { this.fillDuration = value; })
            Text('生成到所选时长，忽略提前EOS').fontSize(13)
          }.width('100%')
        }.padding(16).backgroundColor(Color.White).borderRadius(20)
        Column({ space: 14 }) {
          Text(this.stage).fontSize(21).fontWeight(FontWeight.Medium).width('100%')
          Progress({ value: Math.min(100, this.generatedMs / (this.seconds * 10)), total: 100, type: ProgressType.Linear }).width('100%').color('#258F85')
          Text(`条件帧 ${this.conditionFrames} · 已发布 ${(this.generatedMs / 1000).toFixed(2)} / ${this.seconds} 秒`).fontSize(13).width('100%')
          Row({ space: 12 }) {
            Button(this.busy ? '生成中…' : '生成并播放').enabled(!this.busy && this.modelReady).onClick(() => { this.start(false); }).layoutWeight(1).backgroundColor('#258F85')
            Button('停止生成').enabled(this.busy && !this.stopping).onClick(() => { this.stop(); }).backgroundColor('#E7EFEE').fontColor('#244841')
          }.width('100%')
        }.padding(18).backgroundColor(Color.White).borderRadius(20)
        Column({ space: 12 }) {
          Text(this.waiting ? '正在等待下一段' : this.playing ? '正在播放已生成片段' : '播放器').fontSize(18).width('100%')
          Slider({ value: this.dragging ? this.previewMs : this.playedMs, min: 0, max: Math.max(1, this.generatedMs), step: 20 }).enabled(this.generatedMs > 0)
            .onChange((value: number, mode: SliderChangeMode) => {
              if (mode === SliderChangeMode.Begin) { this.dragging = true; this.previewMs = value; }
              else if (mode === SliderChangeMode.Moving && this.dragging) this.previewMs = value;
              else if (mode === SliderChangeMode.End && this.dragging) { this.dragging = false; this.player?.seek(value); }
              else if (mode === SliderChangeMode.Click) this.player?.seek(value);
            })
          Row() { Text(this.time(this.playedMs)).fontSize(13); Blank(); Text(`可回放至 ${this.time(this.generatedMs)}`).fontSize(13); }.width('100%')
          Button(this.playing ? '暂停播放' : '播放已生成部分').enabled(this.generatedMs > 0).onClick(() => { this.togglePlayback(); })
          Button('系统播放器试听首段').enabled(this.generatedMs > 0).onClick(() => { this.systemListen(); }).backgroundColor('#E7EFEE').fontColor('#244841')
          Text('进度条仅包含已发布音频；停止生成后仍可回放。').fontSize(12).fontColor('#667D80')
          if (this.playbackError.length > 0) { Text(this.playbackError).fontSize(13).fontColor('#AB3535') }
        }.padding(18).backgroundColor(Color.White).borderRadius(20)
        Column({ space: 7 }) {
          Text(`首段 ${this.firstChunk.toFixed(2)}s · 累计 ${this.elapsed.toFixed(2)}s · RSS峰值 ${this.rss.toFixed(0)} MiB`).fontSize(12).fontColor('#667D80')
          Text(`D1 ${this.lm.toFixed(2)}s · D2 ${this.dit.toFixed(2)}s · VAE ${this.vae.toFixed(2)}s`).fontSize(12).fontColor('#667D80')
          Text(`播放队列 ${(this.queueMs / 1000).toFixed(2)}s / 0.12s`).fontSize(12).fontColor('#667D80')
        }.width('100%')
        Button('播放链路自检（合成测试音）').enabled(!this.busy).onClick(() => { this.start(true); }).backgroundColor('#E7EFEE').fontColor('#244841')
        Text('歌词与NPY驱动真实采样。DiT固定2步、CFG1.5；0.6B共驻。RSS上限6GiB。').fontSize(12).fontColor('#667D80')
      }.padding(24).width('100%')
    }.backgroundColor('#F2F6F5').width('100%').height('100%')
  }
}
