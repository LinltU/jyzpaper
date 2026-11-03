import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

// Optional Codex runtime renderer; pass the directory produced by label_v4.py.
const out = path.resolve(process.argv[2] ?? path.dirname(fileURLToPath(import.meta.url)));
const p = JSON.parse(await fs.readFile(path.join(out, "workbook_data.json"), "utf8"));
const wb = Workbook.create();
const summary = wb.worksheets.add("设备汇总");
const windows = wb.worksheets.add("窗口证据");
const methods = wb.worksheets.add("方法说明");
const font = "Microsoft YaHei";
const navy = "#344B64";
const muted = "#607080";
// Excel timestamps have no timezone: preserve the source's local clock fields.
function excelDate(v) {
  return Date.parse(`${v}Z`) / 86400000 + 25569;
}
function col(n) {
  let s = "";
  while (n > 0) { n--; s = String.fromCharCode(65 + n % 26) + s; n = Math.floor(n / 26); }
  return s;
}
function basic(sh, range) {
  sh.showGridLines = false;
  sh.getRange(range).format.font = { name: font, size: 10, color: "#243447" };
  sh.getRange(range).format.verticalAlignment = "center";
}
function header(sh, range) {
  sh.getRange(range).format = {
    fill: navy, font: { name: font, size: 10, color: "#FFFFFF", bold: true },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
  };
}
// Flat window evidence. Statistics are reproducible external analytical outputs.
const wc = p.windows.columns, nr = p.windows.rows.length + 1;
const dateSet = new Set(p.windows.date_columns);
const wr = p.windows.rows.map(r => r.map((v, i) => v !== null && dateSet.has(wc[i]) ? excelDate(v) : v));
windows.getRange(`A1:${col(wc.length)}${nr}`).values = [wc, ...wr];
basic(windows, `A1:${col(wc.length)}${nr}`);
windows.getRange(`A1:${col(wc.length)}${nr}`).format.columnWidth = 19;
windows.getRange(`A2:${col(wc.length)}${nr}`).setNumberFormat("0.000");
windows.getRange(`A1:${col(wc.length)}1`).format.rowHeight = 48;
header(windows, `A1:${col(wc.length)}1`);
const idx = Object.fromEntries(wc.map((v, i) => [v, col(i + 1)]));
for (const c of ["device_id", "window_id", "window_type", "triggered_rules", "reason_cn", "peer_group", "location"]) {
  windows.getRange(`${idx[c]}2:${idx[c]}${nr}`).setNumberFormat("@");
}
windows.getRange("A1:B" + nr).format.columnWidth = 26;
windows.getRange(`${idx.reason_cn}1:${idx.reason_cn}${nr}`).format.columnWidth = 38;
windows.getRange(`${idx.peer_group}1:${idx.peer_group}${nr}`).format.columnWidth = 44;
windows.getRange(`${idx.location}1:${idx.location}${nr}`).format.columnWidth = 67;
for (const c of p.windows.date_columns) {
  windows.getRange(`${idx[c]}2:${idx[c]}${nr}`).setNumberFormat("yyyy-mm-dd hh:mm:ss");
  windows.getRange(`${idx[c]}1:${idx[c]}${nr}`).format.columnWidth = 24;
}
for (const c of ["n_points", "is_candidate", "candidate_rank", "same_direction", "peer_jump_count8",
                 "peer_jump_count12", "rule_A", "rule_B", "rule_C", "rule_D", "has_peer", "residual_jump_count2_5"]) {
  windows.getRange(`${idx[c]}2:${idx[c]}${nr}`).setNumberFormat("0");
}
for (const c of ["high_humidity_fraction", "abs2_residual_fraction", "peer_matched_fraction", "peer_complete_fraction"]) {
  windows.getRange(`${idx[c]}2:${idx[c]}${nr}`).setNumberFormat("0.0%");
}
windows.tables.add(`A1:${col(wc.length)}${nr}`, true, "WindowEvidenceV4");
// Keep identifiers as strings; the renderer needs an explicit full-digit format.
windows.getRange(`A2:A${nr}`).setNumberFormat("0");
windows.freezePanes.freezeRows(1);
windows.freezePanes.freezeColumns(2);
windows.getRange(`A2:${col(wc.length)}${nr}`).conditionalFormats.addCustom(`=$${idx.is_candidate}2=1`, {
  fill: "#FBECEE", font: { color: "#9B3442" },
});

// Formula-linked device review.
basic(summary, "A1:L38");
summary.tabColor = navy;
summary.getRange("A2").values = [["真实挂线数据 v4 标注"]];
summary.getRange("A2").format.font = { name: font, size: 15, bold: true, color: navy };
summary.getRange("A3").values = [[`时间匹配已修正。报告规则重建；本次 ${p.audit.candidate_windows} 个候选，报告为 38 个。原始脚本未找到。`]];
summary.getRange("A3").format.font = { name: font, size: 10, color: muted };
summary.getRange("A5:A8").values = [["原始有效点数"], ["窗口总数"], ["候选窗口数"], ["候选点数"]];
summary.getRange("B5:B8").formulas = [["=SUM(B12:B36)"], ["=SUM(C12:C36)"], ["=SUM(D12:D36)"], ["=SUM(F12:F36)"]];
summary.getRange("D5:D8").values = [["报告候选数"], ["候选窗比例"], ["候选点比例"], ["未评估点数"]];
summary.getRange("E5").values = [[38]];
summary.getRange("E6:E7").formulas = [["=B7/B6"], ["=B8/B5"]];
summary.getRange("E8").values = [[p.audit.unassessed_points]];
summary.getRange("E6:E7").setNumberFormat("0.00%");
summary.getRange("B5:B8").setNumberFormat("#,##0");
summary.getRange("G5").values = [["标签：1=候选，0=规则未触发，-1=未评估"]];
summary.getRange("G6").values = [["红色整窗由其中的触发点决定，不表示每个点都独立异常。"]];
summary.getRange("G7").values = [["逐点原始数据及证据见同目录 device_points_v4.csv。"]];
summary.getRange("G8").values = [["0629、0694 的最后各 1 点已保留，未擅自归为正常。"]];
summary.getRange("G5:L8").merge(true);
summary.getRange("G5:L8").format.wrapText = true;
summary.getRange("A5:L8").format.rowHeight = 28;
summary.getRange("A11:L11").values = [[
  "设备号", "有效点", "窗口数", "候选窗口", "候选窗比例", "候选点", "候选点比例",
  "完整peer比例", "首个观测时间", "最后观测时间", "最大匹配差(分)", "设备位置",
]];
header(summary, "A11:L11");
summary.getRange("A11:L11").format.rowHeight = 34;
const si = Object.fromEntries(p.summary.columns.map((c, i) => [c, i]));
const srows = p.summary.rows.map(r => [
  r[si.device_id], r[si.total_points], null, null, null, null, null,
  r[si.peer_complete_fraction], excelDate(r[si.first_timestamp]), excelDate(r[si.last_timestamp]),
  r[si.max_peer_offset_minutes], r[si.location],
]);
summary.getRange("A12:L36").values = srows;
for (let r = 12; r <= 36; r++) {
  const deviceRange = `'窗口证据'!$${idx.device_id}$2:$${idx.device_id}$${nr}`;
  const candidateRange = `'窗口证据'!$${idx.is_candidate}$2:$${idx.is_candidate}$${nr}`;
  const pointRange = `'窗口证据'!$${idx.n_points}$2:$${idx.n_points}$${nr}`;
  summary.getRange(`C${r}:G${r}`).formulas = [[
    `=COUNTIFS(${deviceRange},A${r})`,
    `=COUNTIFS(${deviceRange},A${r},${candidateRange},1)`,
    `=D${r}/C${r}`,
    `=SUMIFS(${pointRange},${deviceRange},A${r},${candidateRange},1)`,
    `=F${r}/B${r}`,
  ]];
}
summary.tables.add("A11:L36", true, "DeviceSummaryV4");
summary.getRange("A12:A36").setNumberFormat("0");
summary.getRange("A11:A36").format.columnWidth = 23;
summary.getRange("B11:D36").format.columnWidth = 13;
summary.getRange("E11:H36").format.columnWidth = 17;
summary.getRange("I11:J36").format.columnWidth = 24;
summary.getRange("K11:K36").format.columnWidth = 21;
summary.getRange("L11:L36").format.columnWidth = 66;
summary.getRange("B12:D36").setNumberFormat("#,##0");
summary.getRange("F12:F36").setNumberFormat("#,##0");
for (const c of ["E", "G", "H"]) summary.getRange(`${c}12:${c}36`).setNumberFormat("0.0%");
summary.getRange("I12:J36").setNumberFormat("yyyy-mm-dd hh:mm:ss");
summary.getRange("K12:K36").setNumberFormat("0.00");
summary.getRange("A12:L36").format.rowHeight = 23;
summary.getRange("A12:L36").conditionalFormats.addCustom("=$D12>0", { fill: "#FBECEE" });
summary.getRange("A38").values = [["分析结果为离线统计伪标签；未触发不等于确定正常，候选不等于检修确认故障。"]];
summary.getRange("A38").format.font = { name: font, size: 10, color: muted };

// Reproducibility choices and exact field semantics.
const mr = [
  ["项目", "内容"],
  ["来源", "V4_CONSOLIDATED_REPORT (1).pdf 第一部分；device_data_110_220_final.xlsx；设备位置来自 config_25_devices.ini 的 DeviceLocations。"],
  ["本版定位", "报告明确规则与阈值的重建版；缺原始代码，不能声称与原实现完全一致。没有为匹配38个候选而调整阈值。"],
  ["核心参数", "32点无重叠；tail≥2；Huber epsilon=1.5，alpha=1e-4；湿度箱边界0/30/40/50/55/60/65/70/75/80/90/100/101。"],
  ["A", "residual_z_mean_z≥3，abs2_residual_fraction≥0.15，peer_z_mean_z≥3，且两个原带符号窗口均值同号。"],
  ["B", "窗口内peer_delta_jump_z≥8的点数≥2。"],
  ["C", "窗口内至少一点peer_delta_jump_z≥12。"],
  ["D", "has_peer=0，abs2_residual_fraction≥0.15，residual_z_abs_q90≥2，residual_jump_z≥2.5的点数≥2。本数据均有peer，D不适用。"],
  ["最终标签", "A/B/C/D任一触发即为候选；候选窗口全部点映射为1；已评估且未触发为0；仅剩1点的尾部标-1。"],
  ["窗口证据表", "各数值为Python离线统计计算结果；设备汇总计数及比例通过Excel公式引用窗口证据。修改表内证据不会重新训练Huber模型。"],
  ...p.assumptions,
  ["点级文件字段", "device_points_v4.csv：source_excel_row可回查原始行；Timestamp为原始观测时间；previous_timestamp为跳变前点；window_id连接窗口；is_candidate为最终伪标签。"],
  ["窗口字段", "residual_z_mean/peer_z_mean保留方向；相应mean_z为全窗口池证据强度；rule_A～D保留所有触发规则；triggered_rules列出重叠规则。"],
  ["匹配字段", "peer_matches_v4.csv保留每条目标与peer时间、offset_minutes、tolerance_minutes、accepted；自适应容差并不保证严格同步。"],
  ["对照差异", `报告547窗、38候选；本版${p.audit.windows}窗、${p.audit.candidate_windows}候选。原始数据共${p.audit.data_points}点，${p.audit.data_points - p.audit.unassessed_points}点进入窗口，${p.audit.unassessed_points}点保留但未评估。`],
];
methods.getRange(`A1:B${mr.length}`).values = mr;
basic(methods, `A1:B${mr.length}`);
header(methods, "A1:B1");
methods.getRange(`A1:A${mr.length}`).format.columnWidth = 23;
methods.getRange(`B1:B${mr.length}`).format.columnWidth = 117;
methods.getRange(`A2:B${mr.length}`).format.wrapText = true;
methods.getRange(`A2:B${mr.length}`).format.verticalAlignment = "top";
methods.getRange(`A2:B${mr.length}`).format.rowHeight = 42;
for (let i = 1; i < mr.length; i++) {
  if (String(mr[i][1]).length > 110) methods.getRange(`A${i+1}:B${i+1}`).format.rowHeight = 60;
}
methods.getRange("A1:B1").format.rowHeight = 27;
methods.freezePanes.freezeRows(1);
methods.tabColor = "#98A5B2";

wb.recalculate();
const info = await wb.inspect({ kind: "table", range: "设备汇总!A5:E8", include: "values,formulas", tableMaxRows: 5, tableMaxCols: 5, maxChars: 3500 });
console.log(info.ndjson);
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 30 }, maxChars: 2000, summary: "final formula error scan" });
console.log(errors.ndjson);
const previews = [
  ["设备汇总", "A1:L20", "summary_preview.png"],
  ["窗口证据", "A1:I10", "windows_preview.png"],
  ["方法说明", "A1:B12", "methods_preview.png"],
];
for (const [sheetName, range, filename] of previews) {
  const image = await wb.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(path.join(out, filename), new Uint8Array(await image.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(out, "device_nonoverlap_windows_v4.xlsx"));
console.log("Saved device_nonoverlap_windows_v4.xlsx");
