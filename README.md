# 分窗转置卷积的封行交接

合成卷材中，两个来源从相邻读头导出潜变量，交叠的上下文用来定位物理格点。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

`latent` 及各窗口值的轴为行、列、通道。卷积核轴为输入通道、该组内的输出通道、核行、核列。组依输入通道连续划分，输出组保持同序。输入格坐标乘其步长，再加核内坐标乘扩张值和全局 `offset` 得到输出坐标，裁剪为左闭右开的全局框。`owned_rows` 指权属，`read_rows` 是张量中可读的全局区间。有效位为假指缺测，`missing` 区间表示来源明确宣告无潜变量。每个来源的 `through` 是已承诺到的行界，不含该行。值累加采用 float64。
`quantitative` 请求只消费无缺测影响的行。`preview` 可以消费已知贡献与缺口并存的封行。`expected_owners` 公布画幅各行的来源，交换来自其他来源不改变该范围的可封状态。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。

## 流式转置卷积（`streamdeconv` 包）

针对“整幅留在内存等齐才能交图”的瓶颈，提供按全局坐标接收潜变量窗口、
按行封行交接的产品能力，画幅长度不改变工作区上界。

- `geometry.DecodeDescriptor`：画幅绑定的唯一解码单描述，承载通道分组、
  步长、扩张、偏移、裁剪与 `expected_owners` 权属公布；支持域用区间算术
  精确求逆（`row_support_inputs`/`col_support_inputs`），核尺寸与扩张可让
  一个窗口影响远处边缘行，封行依据与实际支持域一致。
- `ops.scatter_window`：只对窗口 `owned_rows` 声明权属的格点做分组转置卷积，
  `read_rows` 里的交叠读上下文不贡献；float64 散射累加进按行持有的槽，
  支持负号撤销以实现承诺前修订替换。
- `engine.StreamDeconv`：窗口流状态机。
  - 来源 `through` 是不含该行的承诺界；只有支持域内所有权属来源都越过承诺界，
    且格点有有效窗口贡献或被显式 `missing` 宣告，输出行才能封行。
  - 空输入可证明为零（结构零）与尚未收到任何贡献（pending）严格区分；
    缺失触及有效输出时该行降级为 `preview`，无缺口行才是可测量的
    `quantitative`，回读带通道顺序、裁剪坐标、封行/可测量事件号与缺口列。
  - 承诺生效后只接受等价重发；异值修订生成 `ConflictRecord`（含影响行）
    并抛 `RevisionConflictError`，已交接像素不变，画幅保持可追查。
  - `max_unsealed_rows` 限制未封行槽位；拥塞抛 `CongestionError` 并在
    `blockage()` 中指明阻塞的来源、潜变量行与承诺界，其他已确定的行照常交接。
- `accounting.ScratchAccount`：按容量口径记账，调用方只读输入与已消费输出
  不计入，额外物化副本计入。`StreamDeconv(stage_windows=False)` 零拷贝引用
  调用方张量，把窗口副本完全移出接收端工作区。

```sh
python demo_stream.py data/capture.json --capacity data/capacity.json
```

`tests/test_capacity.py` 用 `capacity.json` 的 16384×107 规模跑完整窗口流，
断言峰值工作区不超过 `peak_scratch_bytes`（524288 字节）且全部行完成封行。
