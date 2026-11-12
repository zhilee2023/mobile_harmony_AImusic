import { UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class EntryAbility extends UIAbility {
  onCreate(want: Want): void {
    const run = want.parameters?.['streamLabTest'];
    if (typeof run === 'string') { AppStorage.setOrCreate<string>('streamLabAutoRun', run); AppStorage.setOrCreate<boolean>('streamLabSeekCheck', run === 'transport-check'); }
    const model = want.parameters?.['streamLabModel'];
    if (typeof model === 'number') AppStorage.setOrCreate<number>('streamLabModel', model);
  }
  onWindowStageCreate(stage: window.WindowStage): void { stage.loadContent('pages/Index'); }
}
