# 手动签名安装LeVo Harmony v0.1.4

用户要求自己操作电脑界面；不使用computer-use。当前App是**数值与内存诊断版**，不是已训练的音乐生成产品。

## 已构建内容

- 工程：`C:\levo-harmony\app`，独立包名`com.zhilee.levoharmony`。
- 未签名安装包：`C:\levo-harmony\dist\LeVo-Harmony-v0.1.4-unsigned.hap`；用户本地签名包在`app\entry\build\default\outputs\default\entry-default-signed.hap`。
- 测试数据：`C:\levo-harmony\dist\vae-stream-256`，恰好12个文件，约0.36GB。
- 使用已有MNN静态库的本机路径保存在忽略的`app/entry/src/main/cpp/levo-local.cmake`，DevEco后续编译仍会链接同一运行时。它不包含签名凭据，也不修改原CosyVoice工程。

## DevEco中操作

1. 在DevEco Studio选择 **File → Open**，打开`C:\levo-harmony\app`，等待工程同步。
2. 进入 **File → Project Structure → Project → Signing Configs**（不同语言界面可能显示“签名配置”）。
3. 选择`default`产品，使用 **Automatically generate signature / 自动生成签名**。确认当前包名是`com.zhilee.levoharmony`，使用你已经登录的华为开发者账号和已连接手机完成自动签名。
4. 点击 **Apply / OK**，顶部设备列表选择Mate 80 Pro，点击绿色 **Run ▶**。若有账号登录或手机确认，由你本人完成。
5. 安装成功后手机应出现 **LeVo Harmony** 图标，首页显示`v0.1.4`及“10.24秒分块实验”。若构建提示`No signingConfig found for product default`，需要把已生成的default签名关联到default产品；仅生成签名而未关联仍会输出unsigned包。

首页“读取设备内存”还会显示HiDebug提供的App进程RSS/VSS限制与RSS/PSS；数值为0或远大于物理RAM的限额哨兵不作为可用上限，不能将系统可用量当作App无OOM保证。

不要把开发证书、私钥、签名密码或DevEco自动写入的签名配置上传GitHub。当前公开仓库不包含这些文件；继续使用明确文件清单提交。

## 导入数值测试文件

1. 用USB文件传输，把电脑`dist\vae-stream-256`整个文件夹复制到手机“文档”或“下载”。它与此前CLI使用的手机临时目录不同；App不能依赖CLI目录权限。
2. 打开LeVo Harmony，点击 **导入12个测试文件**，在系统文件选择器中打开刚复制的目录，多选其中全部12个文件。
3. 首页应显示`测试文件12/12`。这只表示文件被识别，数值验收看后续结果。
4. 先运行 **VAE CPU High · 三窗数值与内存**。结束后再单独运行Low对照，不同时启动两组。
5. 点击 **导出统一测试报告**，分享`levo-test-report.json`即可。

## 结果如何解释

3个256帧窗口，每块10.24秒，重叠64帧/2.56秒，最终有效25.60秒。只保留当前块和前一块PCM尾部；不把整篇PCM累积在内存。

这组按钮使用真实VAE权重及合成latent，不播放数值夹具，显示模块RTF。v0.1.4另有“三模块”诊断按钮，需预装约1.66GB的W8诊断资产，W4对照另加约0.46GB；本机已部署，路径与详情见THREE_MODULE_VALIDATION.md。该链路的随机音乐头未训练，整链流程/内存通过不代表严格数值或歌曲音质通过；尚无播放器/跨窗长流验收。
