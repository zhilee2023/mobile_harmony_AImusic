"""Create an independent app scaffold from the LeVo Music 1.0.6 build layout."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'music-app'; target=ROOT/'stream-app'
    target.mkdir(exist_ok=True)
    for relative in ['hvigorfile.ts','oh-package.json5','hvigor/hvigor-config.json5','entry/hvigorfile.ts',
        'entry/oh-package.json5','entry/build-profile.json5','entry/src/main/module.json5',
        'entry/src/main/resources/base/profile/main_pages.json','entry/src/main/resources/base/element/color.json',
        'entry/src/main/resources/base/media/app_icon.svg','entry/src/main/cpp/types/libentry/oh-package.json5']:
        dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(base/relative,dest)
    app=target/'AppScope/app.json5';app.parent.mkdir(parents=True,exist_ok=True)
    if not app.exists():app.write_text(json.dumps({'app':{'bundleName':'com.zhilee.levostreamlab','vendor':'zhilee2023',
        'versionCode':100000,'versionName':'0.1.0','icon':'$media:app_icon','label':'$string:app_name'}},indent=2))
    strings=target/'entry/src/main/resources/base/element/string.json'
    if not strings.exists():strings.write_text(json.dumps({'string':[{'name':'app_name','value':'LeVo 流式实验'},{'name':'EntryAbility_label','value':'LeVo 流式实验'},{'name':'EntryAbility_desc','value':'独立数值与播放链路实验'}]},ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['build-profile.template.json5','build-profile.json5']:
        if not (target/name).exists():shutil.copyfile(base/'build-profile.template.json5',target/name)
    # Bind only to the independent new package. Signing material is added by
    # DevEco locally, never copied from an old package or committed.
    print(target)

if __name__=='__main__':main()
