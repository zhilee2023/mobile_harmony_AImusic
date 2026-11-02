import native from 'libentry.so';
import { common } from '@kit.AbilityKit';
import { fileIo as fs, picker } from '@kit.CoreFileKit';
import { media } from '@kit.MediaKit';
import { window } from '@kit.ArkUI';

interface AssetState { ready: boolean; count: number; total: number; bytes: number; }
interface MusicResult { success: boolean; error?: string; cancelled?: boolean; wave?: string; durationSeconds?: number; totalSeconds?: number; wholePhoneRtf?: number; playbackState?: string; playbackValidated?: boolean; }
interface ProgressState { busy: boolean; stage?: string; }

@Entry
@Component
struct Index {
  @State lyrics: string = '[intro-short]\n[verse]\n晚风轻轻吹过窗台\n星光落在你的肩膀\n沿着灯火走向远方\n把心愿唱给月亮\n[chorus]\n让我们迎着风歌唱\n让明天开出新的光\n就算路途还有漫长\n你在身旁就是方向\n[outro-short]';
  @State style: string = 'female, warm clear vocal, Mandarin pop, acoustic guitar, piano, soft drums, hopeful, melodic, 90 BPM';
  @State ready: boolean = false;
  @State busy: boolean = false;
  @State importing: boolean = false;
  @State status: string = '写下歌词，选择属于你的声音。';
  @State modelStatus: string = '正在检查本地模型';
  @State elapsed: number = 0;
  @State audioPath: string = '';
  @State playing: boolean = false;
  @State playPosition: number = 0;
  @State audioDuration: number = 30000;
  @State seconds: number = 30;
  @State resultNote: string = '';
  @State stageIndex: number = 0;
  @State progressValue: number = 0;
  @State progressDetail: string = '';
  private stageLabels: string[] = ['理解歌词与风格', '创作旋律与歌声', '还原音乐细节', '合成与保存音频'];
  private player: media.AVPlayer | undefined = undefined;
  private audioFile: fs.File | undefined = undefined;
  private timer: number = -1;
  private started: number = 0;
  private firstSongCheck: boolean = false;
  private ditSteps: number = 2;
  private ctx(): common.UIAbilityContext { return getContext(this) as common.UIAbilityContext; }
  private root(): string { return `${this.ctx().filesDir}/music-models`; }
  private stamp(value: number): string { const sec = Math.floor(value / 1000); return `${Math.floor(sec / 60)}:${(sec % 60).toString().padStart(2, '0')}`; }
  private refresh(): void {
    try {
      const state = JSON.parse(native.inspect(this.root())) as AssetState;
      this.ready = state.ready;
      this.modelStatus = state.ready ? '已就绪 · 本地生成' : `模型 ${state.count}/${state.total} · 需导入约 ${(state.bytes / 1073741824).toFixed(1)} GB`;
    } catch (_) { this.modelStatus = '请导入模型文件'; }
  }
  aboutToAppear(): void {
    this.refresh();
    try { const last = JSON.parse(fs.readTextSync(`${this.root()}/last-music-result.json`)) as MusicResult;
      if (last.success && last.wave && fs.accessSync(last.wave)) { this.audioPath = last.wave; this.audioDuration = (last.durationSeconds ?? 30) * 1000; }
    } catch (_) {}
    if (AppStorage.get<boolean>('musicAppFirstSongCheck') === true && this.ready) {
      AppStorage.delete('musicAppFirstSongCheck'); this.firstSongCheck = true; this.seconds = 15;
      this.lyrics = '[intro-short]\n[verse]\n清晨的风吹过山岗\n我把梦想装进行囊\n[chorus]\n一路向着阳光歌唱\n你是我心中的方向\n[outro-short]';
      this.style = 'male, warm intimate vocal, Mandarin folk pop, acoustic guitar, gentle piano, light drums, hopeful, 88 BPM';
      setTimeout(() => { this.generate(); }, 500);
    }
    if (AppStorage.get<boolean>('musicAppThirtySecondCheck') === true && this.ready) {
      AppStorage.delete('musicAppThirtySecondCheck'); this.firstSongCheck = true; this.seconds = 30;
      this.lyrics = '[intro-short]\n[verse]\n清晨的风吹过山岗\n我把梦想装进行囊\n[chorus]\n一路向着阳光歌唱\n你是我心中的方向\n[outro-short]';
      this.style = 'male, warm intimate vocal, Mandarin folk pop, acoustic guitar, gentle piano, light drums, hopeful, 88 BPM';
      setTimeout(() => { this.generate(); }, 500);
    }
  }
  aboutToDisappear(): void { this.releasePlayer(); }
  private async keepAwake(enable: boolean): Promise<void> {
    try { const win = await window.getLastWindow(this.ctx()); await win.setWindowKeepScreenOn(enable); } catch (_) {}
  }
  private async importModels(): Promise<void> {
    this.importing = true;
    try {
      const view = new picker.DocumentViewPicker(this.ctx());
      const uris = await view.select({ maxSelectNumber: 30 });
      for (let i = 0; i < uris.length; i++) {
        const uri = uris[i]; const name = decodeURIComponent(uri.substring(uri.lastIndexOf('/') + 1));
        this.status = `正在导入 ${i + 1}/${uris.length}：${name}`;
        const file = fs.openSync(uri, fs.OpenMode.READ_ONLY);
        try { const result = JSON.parse(await native.importAsset(this.root(), name, file.fd)) as MusicResult;
          if (!result.success) throw new Error(result.error ?? '导入失败');
        } finally { fs.closeSync(file); }
      }
      this.status = '模型导入完成';
    } catch (error) { this.status = `导入未完成：${String(error)}`; }
    this.refresh(); this.importing = false;
  }
  private async generate(): Promise<void> {
    if (this.busy || this.importing || !this.ready) return;
    this.busy = true; this.elapsed = 0; this.stageIndex = 0; this.progressValue = 2; this.progressDetail = '正在准备';
    this.started = Date.now(); this.status = '正在准备歌词与风格';
    await this.releasePlayer(); await this.keepAwake(true);
    this.timer = setInterval(() => {
      this.elapsed = Math.floor((Date.now() - this.started) / 1000);
      try { const progress = JSON.parse(native.progress(this.root())) as ProgressState; const stage = progress.stage ?? '';
        if (stage.startsWith('real-phone-music-codes-')) {
          const count = Number(stage.substring('real-phone-music-codes-'.length)); this.stageIndex = 1;
          this.status = '正在创作旋律与歌声'; this.progressDetail = '旋律正在成形';
          this.progressValue = Math.max(this.progressValue, 8 + 52 * count / (this.seconds * 25));
        } else if (stage.startsWith('chunk-')) {
          const parts = stage.split('-'); const chunk = Number(parts[1]);
          const chunks = Math.max(1, Math.ceil((this.seconds * 25 - 256) / 192) + 1);
          const completed = parts[2] === 'wav'; const vocal = parts[2] === 'vae';
          const step = parts[2] === 'dit' ? Number(parts[3]) : this.ditSteps;
          this.stageIndex = vocal || completed ? 3 : 2;
          this.status = this.stageLabels[this.stageIndex]; this.progressDetail = `片段 ${chunk}/${chunks}`;
          this.progressValue = Math.max(this.progressValue, 60 + 39 * ((chunk - 1) + (completed ? 1 : step / (this.ditSteps + 1))) / chunks);
        } else if (stage.includes('prefill') || stage.includes('loaded')) {
          this.stageIndex = 0; this.status = '正在理解歌词与风格'; this.progressDetail = '正在理解你的创作';
          this.progressValue = Math.max(this.progressValue, 5);
        }
      } catch (_) {}
    }, 800);
    try {
      const raw = await native.generate(this.root(), this.lyrics, this.style, this.seconds * 25);
      const result = JSON.parse(raw) as MusicResult;
      if (result.success && result.wave) {
        this.audioPath = result.wave; this.audioDuration = (result.durationSeconds ?? this.seconds) * 1000;
        this.playPosition = 0; this.status = '你的音乐已完成';
        this.progressValue = 100; this.stageIndex = 4; this.progressDetail = '已完成';
        this.resultNote = `${(result.durationSeconds ?? 0).toFixed(1)} 秒音乐 · 用时 ${(result.totalSeconds ?? 0).toFixed(0)} 秒`;
      } else { this.status = result.cancelled ? '本次创作已停止' : `生成未完成：${result.error ?? '请重试'}`; }
    } catch (error) { this.status = `生成未完成：${String(error)}`; }
    if (this.timer >= 0) clearInterval(this.timer); this.timer = -1;
    this.busy = false; await this.keepAwake(false);
    if (this.firstSongCheck && this.audioPath.length > 0) { this.firstSongCheck = false; await this.playPause(); }
  }
  private async releasePlayer(): Promise<void> {
    if (this.player) { try { await this.player.release(); } catch (_) {} this.player = undefined; }
    if (this.audioFile) { fs.closeSync(this.audioFile); this.audioFile = undefined; }
    this.playing = false;
  }
  private async playPause(): Promise<void> {
    try {
      if (this.player?.state === 'playing') { await this.player.pause(); this.playing = false; return; }
      if (this.player?.state === 'paused') { await this.player.play(); this.playing = true; return; }
      await this.releasePlayer(); const player = await media.createAVPlayer(); this.player = player;
      player.on('stateChange', async (state: string) => {
        this.recordPlayback(state);
        try { if (state === 'initialized') await player.prepare();
          if (state === 'prepared') { this.audioDuration = player.duration; await player.play(); }
          this.playing = state === 'playing'; if (state === 'completed') this.playPosition = this.audioDuration;
        } catch (error) { this.status = `播放失败：${String(error)}`; }
      });
      player.on('timeUpdate', (time: number) => { this.playPosition = time; });
      player.on('error', (error) => { this.status = `播放失败：${error.message}`; this.playing = false; });
      this.audioFile = fs.openSync(this.audioPath, fs.OpenMode.READ_ONLY);
      player.fdSrc = { fd: this.audioFile.fd, offset: 0, length: fs.statSync(this.audioPath).size };
    } catch (error) { this.status = `播放失败：${String(error)}`; }
  }
  private recordPlayback(state: string): void {
    try {
      const path = `${this.root()}/last-music-result.json`;
      const result = JSON.parse(fs.readTextSync(path)) as MusicResult;
      if (result.wave !== this.audioPath) return;
      result.playbackState = state;
      if (state === 'playing' || state === 'completed') result.playbackValidated = true;
      const file = fs.openSync(path, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
      try { fs.writeSync(file.fd, JSON.stringify(result)); } finally { fs.closeSync(file); }
    } catch (_) {}
  }
  private async saveFile(report: boolean): Promise<void> {
    try {
      const source = report ? `${this.root()}/last-music-result.json` : this.audioPath;
      const view = new picker.DocumentViewPicker(this.ctx());
      const paths = await view.save({ newFileNames: [report ? 'levo-music-report.json' : 'LeVo-我的音乐.wav'] });
      if (paths.length) { await fs.copyFile(source, paths[0]); this.status = report ? '生成信息已保存' : '音乐已保存'; }
    } catch (error) { this.status = `保存失败：${String(error)}`; }
  }
  build() {
    Scroll() {
      Column({ space: 20 }) {
        Row() {
          Column({ space: 5 }) {
            Text('LeVo 音乐').fontSize(29).fontWeight(FontWeight.Bold).fontColor('#17232E')
            Text('把文字，唱成一首歌').fontSize(14).fontColor('#73808E')
          }.alignItems(HorizontalAlign.Start)
          Blank()
        }.width('100%')
        Column({ space: 12 }) {
          Text('歌词').fontSize(17).fontWeight(FontWeight.Medium).width('100%')
          if (this.busy) {
            Text(this.lyrics.replace(/\n/g, ' ')).fontSize(15).fontColor('#73847A').maxLines(2).textOverflow({ overflow: TextOverflow.Ellipsis }).width('100%')
          } else {
            TextArea({ text: this.lyrics, placeholder: '写下想唱的歌词…' }).height(210).backgroundColor('#F8F9FB').fontSize(15).borderRadius(14)
              .onChange((value: string) => { this.lyrics = value; })
          }
          Text('风格').fontSize(17).fontWeight(FontWeight.Medium).width('100%').margin({ top: 2 })
          if (this.busy) {
            Text(this.style).fontSize(14).fontColor('#73847A').maxLines(2).textOverflow({ overflow: TextOverflow.Ellipsis }).width('100%')
          } else {
            TextArea({ text: this.style, placeholder: '例如：温暖女声、中文流行、钢琴与吉他' }).height(88).backgroundColor('#F8F9FB').fontSize(14).borderRadius(14)
              .onChange((value: string) => { this.style = value; })
          }
          Row({ space: 8 }) {
            Text('时长').fontSize(14).fontColor('#73808E'); Blank()
            Button('15 秒').fontSize(13).height(33).backgroundColor(this.seconds === 15 ? '#E0F0EA' : '#F1F3F5').fontColor('#335E53').enabled(!this.busy).onClick(() => { this.seconds = 15; })
            Button('30 秒').fontSize(13).height(33).backgroundColor(this.seconds === 30 ? '#E0F0EA' : '#F1F3F5').fontColor('#335E53').enabled(!this.busy).onClick(() => { this.seconds = 30; })
          }.width('100%')
        }.padding(18).backgroundColor('#FFFFFF').borderRadius(22).width('100%')
        Button(this.busy ? '创作中…' : '生成音乐').width('100%').height(54).fontSize(17).fontWeight(FontWeight.Medium)
          .backgroundColor('#237C69').enabled(this.ready && !this.busy && !this.importing && this.lyrics.trim().length > 0 && this.style.trim().length > 0)
          .onClick(() => { this.generate(); })
        if (this.busy) {
          Column({ space: 14 }) {
            Row() { Text(this.status).fontSize(19).fontWeight(FontWeight.Medium).fontColor('#245B4D'); Blank()
              Text(`${this.elapsed} 秒`).fontSize(15).fontColor('#7D8B85') }.width('100%')
            Progress({ value: this.progressValue, total: 100, type: ProgressType.Linear }).color('#237C69').backgroundColor('#E3EDE8').width('100%')
              .style({ strokeWidth: 9, enableScanEffect: true, enableSmoothEffect: true })
            ForEach(this.stageLabels, (label: string, index: number) => {
              Row({ space: 10 }) {
                Text(this.stageIndex > index ? '✓' : `${index + 1}`).fontSize(15).fontWeight(FontWeight.Medium)
                  .fontColor(this.stageIndex >= index ? '#237C69' : '#ABB5AE').width(24)
                Text(label).fontSize(17).fontWeight(this.stageIndex === index ? FontWeight.Medium : FontWeight.Normal)
                  .fontColor(this.stageIndex === index ? '#254F42' : '#8A968F'); Blank()
                if (this.stageIndex === index) { LoadingProgress().width(18).height(18).color('#237C69') }
              }.width('100%')
            })
            Row() { Text(this.progressDetail).fontSize(14).fontColor('#7D8B85'); Blank()
              Text('停止').fontColor('#6F8277').fontSize(15).onClick(() => { native.cancel(); this.status = '正在停止…'; }) }.width('100%')
          }.padding(18).backgroundColor('#FFFFFF').borderRadius(20).width('100%')
        } else { Text(this.status).fontSize(17).fontColor('#648073').width('100%') }
        if (this.audioPath.length > 0) {
          Column({ space: 12 }) {
            Row() { Text('我的音乐').fontSize(17).fontWeight(FontWeight.Medium); Blank(); Text('WAV').fontSize(11).fontColor('#88948E') }.width('100%')
            Row({ space: 12 }) {
              Button(this.playing ? '暂停' : '播放').width(68).height(42).backgroundColor('#E5F2ED').fontColor('#237C69').enabled(!this.busy).onClick(() => { this.playPause(); })
              Column({ space: 4 }) {
                Slider({ value: this.playPosition, min: 0, max: this.audioDuration }).selectedColor('#237C69').trackColor('#E7ECE9').blockSize({ width: 10, height: 10 })
                  .onChange((value: number, mode: SliderChangeMode) => { if (mode === SliderChangeMode.End && this.player) this.player.seek(value); })
                Row() { Text(this.stamp(this.playPosition)).fontSize(11).fontColor('#8B9791'); Blank(); Text(this.stamp(this.audioDuration)).fontSize(11).fontColor('#8B9791') }.width('100%')
              }.layoutWeight(1)
            }.width('100%')
            Row() { Text(this.resultNote).fontSize(12).fontColor('#8A958F'); Blank(); Text('保存音频').fontSize(13).fontColor('#237C69').onClick(() => { this.saveFile(false); }) }.width('100%')
          }.padding(18).backgroundColor('#FFFFFF').borderRadius(22).width('100%')
        }
        Column({ space: 9 }) {
          Text(this.modelStatus).fontSize(12).fontColor('#929D96')
          Row({ space: 22 }) {
            Text('导入模型').fontSize(12).fontColor('#6E8276').enabled(!this.busy && !this.importing).onClick(() => { this.importModels(); })
            Text('刷新').fontSize(12).fontColor('#6E8276').onClick(() => { this.refresh(); })
            if (this.audioPath.length > 0) { Text('生成信息').fontSize(12).fontColor('#6E8276').onClick(() => { this.saveFile(true); }) }
          }
        }.width('100%').margin({ top: 2, bottom: 18 })
      }.padding({ left: 22, right: 22, top: 40, bottom: 24 }).width('100%')
    }.width('100%').height('100%').backgroundColor('#F7F8FA')
  }
}
