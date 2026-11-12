import { audio } from '@kit.AudioKit';
import { fileIo as fs } from '@kit.CoreFileKit';

export interface StreamChunk { path: string; startSample: number; samples: number; index: number; }
export interface StreamStatus {
  runId?: string; busy: boolean; status: string; phase?: string; error?: string;
  chunks: StreamChunk[]; generatedSamples: number; requestedFrames: number; conditionFrames?: number;
  lmSeconds?: number; muSeconds?: number; ditSeconds?: number; vaeSeconds?: number;
  elapsedSeconds?: number; firstChunkSeconds?: number; rssPeakMiB?: number;
}
export interface PlaybackStatus { sample: number; playing: boolean; waiting: boolean; submittedAheadSamples: number; error: string; recoveries?: number; }

// One renderer, immutable PCM chunks on disk, at most120ms submitted audio.
// Seek/pause/flush are serialized in the same loop as writes. A request never
// races an outstanding write or reads a file still being generated.
export class StreamPlayer {
  private renderer: audio.AudioRenderer | undefined = undefined;
  private status: StreamStatus = { busy: false, status: 'idle', chunks: [], generatedSamples: 0, requestedFrames: 0 };
  private wanted: boolean = true;
  private disposed: boolean = false;
  private pendingSeek: number = -1;
  private cursor: number = 0;
  private base: number = 0;
  private heard: number = 0;
  private running: boolean = false;
  private file: fs.File | undefined = undefined;
  private filePath: string = '';
  private loop: Promise<void> | undefined = undefined;
  private lastNotice: number = 0;
  private recoveries: number = 0;
  private notify: (state: PlaybackStatus) => void;
  constructor(notify: (state: PlaybackStatus) => void) { this.notify = notify; }
  update(status: StreamStatus): void { this.status = status; }
  play(): void { if (this.heard >= this.status.generatedSamples && !this.status.busy) this.pendingSeek = 0; this.wanted = true; }
  pause(): void { this.wanted = false; }
  seek(milliseconds: number): void {
    const available = this.status.generatedSamples;
    if (available <= 0) return;
    // Published token frames are40ms; use20ms renderer blocks. Seeking to the
    // very last single sample otherwise produces a4-byte device write.
    const lastBlock = Math.floor((available - 1) / 960) * 960;
    this.pendingSeek = Math.max(0, Math.min(lastBlock, Math.floor(milliseconds / 20) * 960));
  }
  private async wait(ms: number): Promise<void> { return new Promise<void>((resolve) => { setTimeout(resolve, ms); }); }
  private announce(waiting: boolean, error: string = ''): void {
    if (Date.now() - this.lastNotice < 100 && error.length === 0) return;
    this.lastNotice = Date.now();
    this.notify({ sample: this.heard, playing: this.running && this.wanted, waiting: waiting,
      submittedAheadSamples: Math.max(0, this.cursor - this.heard), error: error, recoveries: this.recoveries });
  }
  private async position(): Promise<void> {
    if (!this.renderer || !this.running) return;
    const stamp = await this.renderer.getAudioTimestampInfo();
    this.heard = Math.min(this.cursor, this.status.generatedSamples, this.base + Math.max(0, stamp.framePos));
  }
  private async flushAt(sample: number): Promise<void> {
    if (!this.renderer) return;
    if (this.running) await this.renderer.pause();
    this.running = false;
    if (this.renderer.state === audio.AudioState.STATE_PAUSED || this.renderer.state === audio.AudioState.STATE_RUNNING) await this.renderer.flush();
    this.cursor = sample; this.base = sample; this.heard = sample;
  }
  async open(): Promise<void> {
    await this.openRenderer(); this.loop = this.pump();
  }
  private async openRenderer(): Promise<void> {
    this.renderer = await audio.createAudioRenderer({ streamInfo: {
      samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_48000, channels: audio.AudioChannel.CHANNEL_2,
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE, encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW },
      rendererInfo: { usage: audio.StreamUsage.STREAM_USAGE_MUSIC, rendererFlags: 0 } });
    await this.renderer.setVolume(1);
  }
  private async recoverRenderer(): Promise<void> {
    const renderer = this.renderer; this.renderer = undefined; this.running = false;
    if (renderer) { try { await renderer.stop(); } catch (_) {} try { await renderer.release(); } catch (_) {} }
    this.cursor = this.heard; this.base = this.heard;
    await this.openRenderer();
  }
  private async pump(): Promise<void> {
    try {
      while (!this.disposed) {
        try {
        if (!this.renderer) return;
        await this.position();
        if (this.pendingSeek >= 0) {
          const target = Math.max(0, Math.min(this.pendingSeek, this.status.generatedSamples - 1));
          this.pendingSeek = -1; await this.flushAt(target);
        }
        if (!this.wanted) {
          if (this.running) await this.flushAt(this.heard);
          this.announce(false); await this.wait(30); continue;
        }
        if (this.cursor >= this.status.generatedSamples) {
          if (this.running) {
            await this.renderer.drain();
            await this.flushAt(this.cursor);
          }
          if (!this.status.busy && this.status.generatedSamples > 0) this.wanted = false;
          this.announce(this.status.busy); await this.wait(20); continue;
        }
        if (!this.running) { await this.renderer.start(); this.running = true; }
        if (this.cursor - this.heard > 4800) { this.announce(false); await this.wait(10); continue; }
        let found: StreamChunk | undefined = undefined;
        for (const chunk of this.status.chunks) {
          if (this.cursor >= chunk.startSample && this.cursor < chunk.startSample + chunk.samples) { found = chunk; break; }
        }
        if (!found) throw new Error('已生成音频目录存在空洞');
        if (this.filePath !== found.path) {
          if (this.file) fs.closeSync(this.file);
          this.file = fs.openSync(found.path, fs.OpenMode.READ_ONLY); this.filePath = found.path;
        }
        if (!this.file) throw new Error('无法读取已生成片段');
        const samples = Math.min(960, found.startSample + found.samples - this.cursor, this.status.generatedSamples - this.cursor);
        const buffer = new ArrayBuffer(samples * 4);
        fs.lseek(this.file.fd, 44 + (this.cursor - found.startSample) * 4, fs.WhenceType.SEEK_SET);
        const count = fs.readSync(this.file.fd, buffer);
        if (count !== buffer.byteLength) throw new Error('音频片段未完整发布');
        let written = 0;
        let retries = 0;
        while (written < count) {
          const bytes = await this.renderer.write(written === 0 ? buffer : buffer.slice(written));
          if (bytes === 0 && retries++ < 100) { await this.wait(5); continue; }
          if (bytes <= 0 || bytes % 4 !== 0) throw new Error(`音频写入失败：${bytes}/${count - written} 字节`);
          written += bytes; this.cursor += bytes / 4;
        }
        this.announce(false);
        } catch (error) {
          if (!this.disposed && String(error).includes('system error') && this.recoveries < 3) {
            this.recoveries++; await this.recoverRenderer(); this.announce(false); continue;
          }
          throw new Error(String(error));
        }
      }
    } catch (error) { this.running = false; this.notify({ sample: this.heard, playing: false, waiting: false, submittedAheadSamples: 0, error: String(error) }); }
  }
  async close(): Promise<void> {
    this.disposed = true;
    if (this.loop) await this.loop;
    if (this.renderer) { try { await this.renderer.stop(); } catch (_) {} await this.renderer.release(); this.renderer = undefined; }
    if (this.file) { fs.closeSync(this.file); this.file = undefined; }
  }
}
