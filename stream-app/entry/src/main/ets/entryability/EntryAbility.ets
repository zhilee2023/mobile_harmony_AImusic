import { UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class EntryAbility extends UIAbility {
  onCreate(want: Want): void {
    const run = want.parameters?.['streamLabTest'];
    if (typeof run === 'string') { AppStorage.setOrCreate<string>('streamLabAutoRun', run); AppStorage.setOrCreate<boolean>('streamLabSeekCheck', run === 'transport-check'); }
    const model = want.parameters?.['streamLabModel'];
    if (typeof model === 'number') AppStorage.setOrCreate<number>('streamLabModel', model);
    const seconds = want.parameters?.['streamLabSeconds'];
    if (typeof seconds === 'number') AppStorage.setOrCreate<number>('streamLabSeconds', seconds);
    const dynamic = want.parameters?.['streamLabDynamicOption'];
    if (typeof dynamic === 'number') AppStorage.setOrCreate<number>('streamLabDynamicOption', dynamic);
    const tag = want.parameters?.['streamLabRunTag'];
    if (typeof tag === 'string') AppStorage.setOrCreate<string>('streamLabRunTag', tag);
    const fusion = want.parameters?.['streamLabVaeFusion'];
    if (typeof fusion === 'number') AppStorage.setOrCreate<boolean>('streamLabVaeFusion', fusion === 1);
  }
  onWindowStageCreate(stage: window.WindowStage): void { stage.loadContent('pages/Index'); }
}
