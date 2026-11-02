import { UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class EntryAbility extends UIAbility {
  onCreate(want: Want): void {
    if (want.parameters?.['musicAutoRun'] === 'llm-ui-aba-20261102') AppStorage.setOrCreate<boolean>('musicAppUiBenchmark', true);
    if (want.parameters?.['musicAutoRun'] === 'first-song-20261102') AppStorage.setOrCreate<boolean>('musicAppFirstSongCheck', true);
    if (want.parameters?.['musicAutoRun'] === 'progress-song-30s-20261102') AppStorage.setOrCreate<boolean>('musicAppThirtySecondCheck', true);
  }
  onWindowStageCreate(stage: window.WindowStage): void { stage.loadContent('pages/Index'); }
}
