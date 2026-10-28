import native from 'libentry.so';

@Entry
@Component
struct Index {
  @State lyrics: string = '[Verse]\n晚风轻轻吹过窗台\n星光落在你的眼海';
  @State style: number = 0;
  @State status: string = 'v0.1 · 架构与内存检查；尚无训练好的0.6B音乐权重';
  @State memory: string = '';
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
        Text('歌词 → 1个风格前缀＋Qwen3骨干 → 混合音乐Token → DiT → 48kHz双声道VAE')
          .fontColor('#CCD7E8').fontSize(15)
        Text('Qwen文本预训练权重可加载；音乐embedding、输出头和风格向量必须训练。此版本不显示虚假的生成/播放结果。')
          .fontColor('#A9B9D2').fontSize(13)
      }.padding(24).width('100%')
    }.width('100%').height('100%').backgroundColor('#101A2C')
  }
}
