import { UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class EntryAbility extends UIAbility {
  onCreate(want: Want): void {
    const precision = want.parameters?.['levoDiagnostic'];
    if (precision === 'chain-comparison-large-2' || precision === 'chain-comparison-small-2' || precision === 'high' || precision === 'low' || precision === 'chain-serial' ||
      precision === 'chain-keep-llm' || precision === 'chain-all' || precision === 'chain-all-w4' ||
      precision === 'chain-llm-short' || precision === 'chain-llm-long' || precision === 'chain-chunks-5' || precision === 'chain-chunks-5-tuned' || precision === 'chain-chunks-5-w8-vsin' || precision === 'chain-chunks-5-w8-snake') {
      AppStorage.setOrCreate<string>('levoDiagnosticPrecision', precision as string);
      const runId = want.parameters?.['levoDiagnosticRunId'];
      if (typeof runId === 'string') AppStorage.setOrCreate<string>('levoDiagnosticRunId', runId);
    }
  }
  onWindowStageCreate(stage: window.WindowStage): void { stage.loadContent('pages/Index'); }
}
