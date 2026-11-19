# 从GitHub拉取后，在另一台手机安装

`main`里有两个独立工程：[music-app](../music-app/)是LeVo音乐1.0.6，
[stream-app](../stream-app/)是本轮优化的LeVo流式实验。
请在DevEco打开所需子目录，不把整个仓库根目录当成单个Harmony工程。

## 为什么拉取后没有现成签名

两款App真实的`build-profile.json5`及`.p12/.cer/.p7b`均被Git忽略；仓库只保留
`build-profile.template.json5`。本地签名配置含电脑上的证书路径和签名材料，不能作为所有人的通用配置上传。
调试Profile还包含包名、证书和允许调试的设备列表；旧手机的Profile可能不包含新手机。
这与模型是否存在是两个不同步骤。

华为说明：[自动签名](https://developer.huawei.com/consumer/cn/doc/doccenter-deveco-studio/ide-signing-auto)、
[Profile与设备UDID／安装失败](https://developer.huawei.com/consumer/cn/doc/harmonyos-faqs-V2/faq-debugging-and-running-0000001166752005-V2/)。
没有具体错误日志时，不能把所有编译或安装失败都归因于签名。

## 拉取后，改成可与原版并存的新App

克隆到一个新目录，初始化时同时指定**新包名**和**显示名称**：

```powershell
git clone https://github.com/zhilee2023/mobile_harmony_AImusic.git my-music-app
cd my-music-app
powershell -NoProfile -File tools/init_harmony_app.ps1 -App stream-app `
  -BundleName com.yourname.mystreammusic `
  -DisplayName "我的音乐实验"
```

`com.yourname.mystreammusic`是示例，替换成自己的唯一包名。脚本会：

- 从模板生成DevEco识别工程所需的本地`build-profile.json5`。
- 更新`AppScope/app.json5`中的`bundleName`。
- 更新桌面名称、Ability标签及各语言的`app_name`资源；页面标题也使用这个资源。

随后在DevEco打开脚本输出的`stream-app`目录，连接目标手机，
进入Signing Configs，用自己的开发者账号为**新包名**自动签名，再点击Run安装。
新包名对应独立App和数据目录，原版可以保留；模型和歌曲不会自动跨包名共享。

只改文件夹名或显示名称，不会创建新的应用身份；要与原版并存必须改包名。
新包名不能直接使用旧包名的调试Profile，自动签名需要重新生成。
为防止误改原项目，脚本遇到已有非空签名配置时会拒绝更换包名，请在全新克隆目录中执行。
仓库里的默认包名不会因别人运行该命令而改变；更名只作用于该克隆。

构建／部署方式不变。`tools/deploy_stream_app.ps1`与`tools/run_music_duration.ps1`
会读取工程的新包名来启动App、定位数据目录和检查前台；其他历史研究脚本仍需核对其固定包名后再使用。

## 同一台电脑，换一台手机

1. 用USB连接新手机，开启开发者选项及USB调试，并允许这台电脑调试。
2. 在仓库根目录初始化所选工程。该步骤无需SDK或MNN，只在文件不存在时创建配置，保留已有签名。

```powershell
powershell -NoProfile -File tools/init_harmony_app.ps1 -App music-app
# 本轮流式实验改用：-App stream-app
```

3. DevEco打开脚本输出的`music-app`或`stream-app`完整目录，等待同步完成；不要打开`entry`。
   若提示“不包含App／不是有效工程”，先核对该目录下的`build-profile.json5`已生成。
4. 进入`File → Project Structure → Project → Signing Configs`，选择`default`产品，
   用自己的华为开发者账号登录并勾选`Automatically generate signature`。
   确认新手机已被DevEco识别后Apply／OK；换设备后已勾选但仍使用旧Profile时，重新触发自动生成。
5. 在运行目标中选择新手机，点击Run，重新构建并安装该手机可用的已签名HAP。

只换手机通常无需修改包名。同一份调试包不是任意手机都可安装；
若需多台手机调试，Profile必须涵盖这些设备，或按官方流程注册设备并申请相应Profile。

## 同时换电脑／全新拉取

签名之外，还需安装匹配工程的DevEco/Harmony SDK，并准备兼容的MNN源码和
**HarmonyOS ARM64静态库`libMNN.a`**；Windows版MNN或其他平台库不能替代它。
这些本机依赖和SDK配置不在Git中。代码内的本地MNN路径可通过参数指定，不应复制他人的签名路径。

流式实验准备配置，不执行构建：

```powershell
pwsh -File tools/build_stream_app.ps1 -PrepareOnly `
  -MnnRoot C:/your-MNN-source `
  -MnnLibrary C:/your-Harmony-arm64/libMNN.a
```

然后在DevEco中配置新手机的自动签名。构建时去掉`-PrepareOnly`。
`tools/build_music_app.ps1`同样接受`-MnnRoot`和`-MnnLibrary`，首次构建会创建本地配置。
两脚本均保留已存在的签名配置。更名命令应在自动签名之前执行。

全新手机若只安装流式App，可在签名构建成功后运行：

```powershell
pwsh -File tools/deploy_stream_app.ps1 -Device YOUR_DEVICE_ID -InstallOnly
```

这不会尝试从新手机上尚不存在的LarkStudio App复制模型。
原部署脚本默认的模型迁移适用于**同一台手机已有原LarkStudio模型**的情况，不能将其复制步骤当作新手机的自动下载功能。

## 安装后还需要模型

GitHub不含模型权重、原手机App数据、MuQ风格NPY或生成歌曲。
需要另行部署匹配该App合同的完整模型包及风格文件；0.6B和4B的LM／条件投影接口也不同。
空App安装成功，不代表已有生成所需权重。替换模型另见[部署指南](MODEL_DEPLOYMENT.md)。

## 按报错区分

| 现象 | 优先检查 |
|---|---|
| 打开stream提示“不包含App／不是有效工程” | 运行初始化脚本后打开`stream-app`；检查`build-profile.json5`、`AppScope/app.json5`、`entry`、`oh-package.json5`及`hvigorfile.ts`都在同一工程层级 |
| 缺少`build-profile.json5` | 从所选App的模板创建本地文件 |
| 证书／Profile路径不存在、过期或包名不符 | 在本机、正确子工程重新生成签名；不要复制其他工程的配置 |
| `INSTALL_FAILED_APP_SOURCE_NOT_TRUSTED` | 调试Profile是否包含新手机UDID；重新签名并重建HAP |
| 提示与已安装应用签名不一致 | 保持原证书更新，或先导出歌曲／模型后选择卸载重装；卸载会清除该App数据 |
| 缺少MNN头文件／静态库或CMake失败 | 配置兼容MNN路径，与签名分开排查 |
| App可打开但提示模型未就绪 | 单独迁移并校验模型及NPY |

若换自己的开发者账号出现包名注册冲突，可改用自己的包名再重新申请Profile；
同时更新涉及包名的HDC脚本。仅修改显示名称不会改变包名或签名身份。
