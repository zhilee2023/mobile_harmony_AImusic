import { UIAbility, Want, AbilityConstant } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { fileIo as fs } from '@kit.CoreFileKit';
export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launch: AbilityConstant.LaunchParam): void {
    try {
      const file = fs.openSync(`${this.context.filesDir}/last-launch.json`, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
      try { fs.writeSync(file.fd, JSON.stringify(launch)); } finally { fs.closeSync(file); }
    } catch (_) {}
    const precision = want.parameters?.['levoDiagnostic'];
    if (precision === 'chain-comparison-large-2' || precision === 'chain-comparison-small-2' || precision === 'chain-comparison-large-2-60' || precision === 'chain-comparison-small-2-60' || precision === 'chain-comparison-large-2-120' || precision === 'chain-comparison-small-2-120' || precision === 'chain-comparison-large-2-120-low' || precision === 'chain-comparison-large-codec-120' || precision === 'chain-comparison-large-2-90' || precision === 'chain-comparison-small-2-90' || precision === 'chain-comparison-small-vae-90' || precision === 'chain-comparison-large-codec-90' || precision === 'high' || precision === 'low' || precision === 'chain-serial' ||
      precision === 'chain-keep-llm' || precision === 'chain-all' || precision === 'chain-all-w4' ||
      precision === 'chain-llm-short' || precision === 'chain-llm-long' || precision === 'chain-chunks-5' || precision === 'chain-chunks-5-tuned' || precision === 'chain-chunks-5-w8-vsin' || precision === 'chain-chunks-5-w8-snake') {
      AppStorage.setOrCreate<string>('levoDiagnosticPrecision', precision as string);
      const runId = want.parameters?.['levoDiagnosticRunId'];
      if (typeof runId === 'string') AppStorage.setOrCreate<string>('levoDiagnosticRunId', runId);
      const bundle = want.parameters?.['levoModelBundle'];
      if (typeof bundle === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$/.test(bundle) && !bundle.includes('..')) AppStorage.setOrCreate<string>('levoModelBundle', bundle);
    }
  }
  onWindowStageCreate(stage: window.WindowStage): void { stage.loadContent('pages/Index'); }
}
