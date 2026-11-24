import { UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class EntryAbility extends UIAbility {
  onCreate(want: Want): void {
    const mode = want.parameters?.['operatorLabTest'];
    if (typeof mode === 'string' && (mode === 'baseline' || mode === 'operators')) {
      AppStorage.setOrCreate<string>('operatorLabMode', mode);
      const model = want.parameters?.['operatorLabModel'];
      AppStorage.setOrCreate<number>('operatorLabModel', model === 1 ? 1 : 0);
      const steps = want.parameters?.['operatorLabSteps'];
      AppStorage.setOrCreate<number>('operatorLabSteps', typeof steps === 'number' ? steps : 64);
      const rounds = want.parameters?.['operatorLabRounds'];
      AppStorage.setOrCreate<number>('operatorLabRounds', typeof rounds === 'number' ? rounds : 3);
      const threads = want.parameters?.['operatorLabThreads'];
      AppStorage.setOrCreate<number>('operatorLabThreads', typeof threads === 'number' ? threads : 4);
      const precision = want.parameters?.['operatorLabPrecision'];
      AppStorage.setOrCreate<string>('operatorLabPrecision', precision === 'low' ? 'low' : 'high');
      const tag = want.parameters?.['operatorLabRunTag'];
      AppStorage.setOrCreate<string>('operatorLabRunTag', typeof tag === 'string' ? tag : '');
    }
  }
  onWindowStageCreate(stage: window.WindowStage): void { stage.loadContent('pages/Index'); }
}
