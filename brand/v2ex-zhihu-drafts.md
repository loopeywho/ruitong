# V2EX / Zhihu drafts (Chinese)

Both include the Ascend collaboration ask, since this audience is the one most likely to actually
have 910B access. Same real numbers as the English drafts — kept consistent across languages.

---

## V2EX (v2ex.com — post to a technical node, e.g. 分享创造 or 程序员)

**标题：** 同一个模型，同一份权重，两块 NVIDIA 显卡跑出不同答案的概率是 26%——实测，不是估计

**正文：**

一直看到有人说"换加速卡会让 LLM 输出发生一点变化"，但从没见过具体数字，所以自己写了个工具去实测（黑盒、走 HTTP，不需要碰到底层硬件）。

同一个 `Qwen3-8B`、同一个随机种子、同一个 vLLM 镜像，几组加速卡对比下来：

- NVIDIA A40 对 RTX 6000 Ada（同厂商）：61 条提示词里 **26%** 输出文本不同
- NVIDIA A40 对 H100（同厂商）：**25%**
- NVIDIA A40 对 AMD MI300X（跨厂商）：**29.5%**
- 同一块 A100，只是 fp16 换成 bf16，硬件完全没变：**29.5%**

最后一条比较有意思：仅仅是精度不同，产生的分叉幅度就和换厂商差不多。如果你在用同厂商配对当作"正确迁移"的基准，这个基准本身就不是零噪声的。

完整方法、原始 JSON 语料、以及门限的故障注入校准过程都发布在这里：https://ruitong.io ——参考表完全免费，不需要注册。

**另外**：如果有人手上有华为昇腾 910B、跑得起 vLLM 或兼容 OpenAI 接口的推理服务，很想听听你们的情况。不需要开放任何访问权限——在自己本地端点跑一下我们公开的采集脚本，把生成的 JSON 发回来就行，没有任何专有数据或凭证会被暴露。贡献者会署名致谢。

欢迎拍砖，尤其是关于指标设计的部分（用 token 身份对齐比较概率，而不是余弦相似度——余弦对缩放误差不敏感，完全测不出来）。

---

## Zhihu (知乎 — post as a 专栏 article, or as an answer to a relevant question like "如何评估大模型跨硬件迁移的一致性?")

**标题：** 没人愿意公布的数字：换一块显卡，大模型的回答会变多少？我们实测了

**正文：**

如果你正在把 LLM 推理从一种加速卡迁移到另一种——NVIDIA 换 AMD，或者换华为昇腾——迟早会有人问你一个问题：模型的行为到底变了多少？

这个问题目前没有公开的参考答案。华为的 msprobe 只做算子级别的张量比对；MLCommons 只公布了少数几个基准模型的 99%/99.9% FP32 达标率；vLLM 自己的昇腾精度 CI 用的是"允许 5% 漂移"的硬编码阈值，甚至没有 GPU 基线做对照。几乎每一份工具文档都告诉你"跑个精度测试"，但没有一份告诉你应该拿什么数字去比较。

我们决定自己测一下，用真实硬件、真实模型、不用任何合成数据。

**方法**：黑盒对比，走 OpenAI 兼容接口，harness 完全看不到底层硬件，只处理 HTTP 返回的 logprobs。按 token 身份对齐比较概率，而不是按排名比较（top-k 里经常有并列的 token，极小的扰动就能让顺序互换，排名对比会比较错对象）。判定用复合门限：top-1 一致率、概率质量偏差、token 对齐概率差——三项任意一项不达标就判定为不通过，门限本身是用故障注入校准出来的，不是拍脑袋定的。

**结果**：

同一份 `Qwen3-8B`、同一个随机种子，61 条提示词（中英文都有）：

| 对比 | 输出文本不同的比例 |
|---|---|
| NVIDIA A40 vs RTX 6000 Ada（同厂商） | 26% |
| NVIDIA A40 vs H100（同厂商） | 25% |
| NVIDIA A40 vs AMD MI300X（跨厂商） | 29.5% |
| 同一块 A100，仅 fp16 换 bf16（无硬件变化） | 29.5% |

最后一行是我认为最值得关注的：**仅仅是服务精度不同（fp16 vs bf16），产生的分叉幅度就已经接近换一个完全不同的芯片厂商**。也就是说，你以为的"硬件噪声"，很大一部分可能只是精度问题。

完整的方法论、原始语料（可复现、可离线重新分析）、以及门限校准过程发布在 https://ruitong.io 。

**如果你有华为昇腾 910B 的访问权限**：我们非常希望能补上这一行数据，但不需要你开放任何基础设施访问权限——只需要在自己的环境里跑一下我们已经公开的采集脚本（对着本地端点），把生成的 JSON 结果发回来即可。返回内容里不含任何专有数据或凭证，贡献者会在报告中署名致谢。欢迎私信或通过网站联系。

---

## Notes

- V2EX audience wants terse, no-fluff, data-first — kept short.
- Zhihu rewards a fuller narrative/explanation of the "why" — used the longer form, closer to the
  English methodology section.
- Both include the Ascend ask since this is the highest-relevance audience for it — no need to
  post it as a separate item.
