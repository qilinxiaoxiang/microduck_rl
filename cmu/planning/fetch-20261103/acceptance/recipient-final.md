# 独立收件人最终验收

结论：通过任务交接可理解性、干净环境 CPU 启动和已测软件协议验收。原始拒绝记录完整保留；不代表实机自主取球成功。

- 干净副本、全新 Python 3.12 环境和固定依赖沿用；修订前后 requirements 哈希相同。候选从未被本验收者编辑。
- 原始 check_all.py 六项再次全部通过；修订演练 22/22。
- 逐项另跑 A parser/事件输出/源文件校验，B 图像 detector/原 gaze 测试，C/D fixtures/locomotion/pickup，E 默认/新图像轨迹演练，均 exit 0；pip check 无冲突。
- 独立组合反例 7/7：原始 R1 精确序列、有效看人后 R1、返程 R1，以及缺失/错人/过期/有效看人确认。
- R1 闭环：原始序列因缺少看人确认而不再进入接近；补入有效看人确认后，在干扰人物新帧到达且原目标已过期时明确 controlled_stop，出程/返程都正确。
- 看人门槛：settled 只发 head-only gaze；正确原人、新鲜且对齐的 caller_looked 到达后才可找球；错人/旧图像/缺失反馈均保持等待。
- 当前仅验收本地确定性协议和历史推理，不含真实音频/身份、新背包仿真训练、真实 sensor/readback 与保护下实机连续试验。

A–E 的目的、改动、第一步、输出、验收和依赖详见保留的 recipient.md；修订后的协议额外要求先确认看向原人。中文总览能先说明完整行为、五项职责和共同身体基线；任务边界、独立开工与最终连续联测关系清楚。

完整证据：recipient-final.json、recipient-final.inputs.json、recipient-final.counterexamples.json、recipient-final.commands.json、recipient-final.outputs/ 和 recipient-final.*.stdout.txt。输入包括所有实际测试代码/模型/数据哈希。
