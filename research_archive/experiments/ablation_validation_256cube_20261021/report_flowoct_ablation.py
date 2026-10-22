"""Focused native-pixel comparisons of raw FM, Haar FM, and full FlowOCT."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import numpy as np
from engine import layer_projection, load_samples, write_json
from report256 import strip

METHODS = [('07_flowoct_fm_only', 'FlowOCT FM only'),
           ('08_flowoct_wavelet_only', 'FlowOCT Haar only'),
           ('06_flowoct', 'FlowOCT full')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    reports = ROOT / 'reports'
    samples = load_samples(ROOT, read(ROOT / 'data/selection.json'))
    rows, curves, figures = [], [], []
    for arm, label in METHODS:
        out = ROOT / arm / 'one_case'
        for mode, filename in [('native3', 'final_native3.json'), ('shared8', 'final_metrics.json')]:
            result = read(out / filename)
            row = {'arm': arm, 'method': label, 'solver': mode, 'updates': 400}
            for phase in ('train', 'validation'):
                for key in ('ilm_opl_pcc', 'ilm_opl_ssim', 'volume_mae', 'volume_psnr'):
                    row[phase + '_' + key] = result['summary'][phase]['metrics'][key]
            rows.append(row)
        for line in (out / 'history.jsonl').read_text().splitlines():
            record = json.loads(line)
            curves.append({'method': label, 'update': record['update'],
                           'train_pcc': record['summary']['train']['metrics']['ilm_opl_pcc'],
                           'train_ssim': record['summary']['train']['metrics']['ilm_opl_ssim'],
                           'validation_pcc': record['summary']['validation']['metrics']['ilm_opl_pcc'],
                           'validation_ssim': record['summary']['validation']['metrics']['ilm_opl_ssim']})
    for sample in samples:
        truth, _ = layer_projection(sample['octa'], sample['layers'])
        name = sample['case_id'] + '_' + sample['patch_id'] + '.npz'
        for mode, folder in [('native3', 'predictions_native3'), ('shared8', 'predictions')]:
            maps = [('Real OCTA', truth)]
            for arm, label in METHODS:
                with np.load(ROOT / arm / 'one_case' / folder / name, allow_pickle=False) as saved:
                    assert np.array_equal(saved['octa_gt'], sample['octa'])
                    projection, _ = layer_projection(saved['octa'], sample['layers'])
                maps.append((label, projection))
            filename = 'flowoct_ablations_' + sample['case_id'] + '_' + mode + '.png'
            strip(maps, reports / filename, sample['case_id'] + ' ' + sample['split'] +
                  ' | 400 updates | ' + mode + ' | ILM-OPL | same [0,1] window')
            figures.append(filename)
            if sample['split'] == 'train':
                lowfile = 'flowoct_ablations_' + sample['case_id'] + '_' + mode + '_lowwindow.png'
                strip(maps, reports / lowfile, sample['case_id'] +
                      ' TRAIN | 400 updates | ' + mode + ' | same [0,0.25] window including truth', maximum=.25)
                figures.append(lowfile)
    write_json(reports / 'flowoct_ablation_summary.json', rows)
    write_json(reports / 'flowoct_ablation_learning.json', curves)
    with (reports / 'flowoct_ablation_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    notes = [
        '# FlowOCT：仅流匹配 / 仅小波 / 完整辅助监督', '',
        '两组新增实验均在已有中央256³冻结数据上从头训练400次：训练病例10301，四位验证患者不参与训练。',
        '保留统一 /255、原始横向网格、无增强的输入协议；本轮隔离模块变化，没有切回作者的在线预处理。', '',
        '| 组别 | 输入表示 | 训练损失 | 对比头 |', '|---|---|---|---|',
        '| 07 仅流匹配 | 原像素，1通道，无小波 | 10 × 原作者ROI速度MSE | 无 |',
        '| 08 仅小波 | 二维Haar四子带 | 10 × 原作者ROI速度MSE | 无 |',
        '| 06 完整FlowOCT | 二维Haar四子带 | 10速度 + 0.5重建 + 0.005对比 + 0.1投影 | 有 |', '',
        '这里的“仅流匹配”保留作者速度损失的预测/目标包围框并集裁剪规则；不是额外引入新的前景损失。',
        '原作者公开主干为二维网络，逐B-scan处理；这里的小波也是二维Haar，与前五组的三维Haar不同。',
        '三组均没有独立拼接固定OCT条件。训练每次抽64张完整B-scan、每片独立采样t，Adam2e-4、FP32、无梯度裁剪。',
        '新增两组实际抽片和时间逐值保存并比较；采用与既有06相同的更新种子和抽样算法。',
        '08与06在相同种子下的初始生成器权重相同，已用单元检查确认；08删除了对比头和三项辅助损失。', '',
        '07的B-scan网络输入为256×256，08/06的四子带输入为128×128。07仅将输入/输出通道从4改为1，保留主干其他默认设置。',
        '因此07与08同时包含小波表示、网络内部空间尺度和少量参数数量差别，未做计算量匹配。',
        '07按8片及4片执行的显存预检未通过，正式执行改为2片重算、推理4片；08/06训练8片重算、推理16片。每次更新仍完整监督64片。', '',
        '## 400次更新后的结果', '',
        '以下PCC/SSIM均为同一ILM–OPL最大投影；验证按患者等权平均。', '',
        '| 方法 | 推理 | 训练PCC | 训练SSIM | 验证PCC | 验证SSIM |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        notes.append('| ' + row['method'] + ' | ' + row['solver'] + ' | ' + ' | '.join(
            f"{row[phase + '_' + key]:.4f}" for phase in ('train', 'validation')
            for key in ('ilm_opl_pcc', 'ilm_opl_ssim')) + ' |')
    notes += ['', 'native3是三步Euler、每步clamp；shared8是八步Euler、仅终点clamp。两者也存在裁剪策略差异，不能只归因于步数。',
              '两种推理都只接收OCT，真实OCTA只用于训练目标及推理后的评价。', '',
              '## 训练过程：统一八步评价', '',
              '| 方法 | 更新 | 训练PCC | 训练SSIM | 验证PCC | 验证SSIM |', '|---|---:|---:|---:|---:|---:|']
    for row in curves:
        notes.append('| ' + row['method'] + ' | ' + str(row['update']) + ' | ' + ' | '.join(
            f'{row[key]:.4f}' for key in ('train_pcc', 'train_ssim', 'validation_pcc', 'validation_ssim')) + ' |')
    shared = {row['arm']: row for row in rows if row['solver'] == 'shared8'}
    raw, haar, full = [shared[arm] for arm, _ in METHODS]
    notes += ['', '## 本轮观察', '',
              f"无小波、仅速度匹配组的八步训练PCC为{raw['train_ilm_opl_pcc']:.4f}，验证PCC为{raw['validation_ilm_opl_pcc']:.4f}；"
              f"仅小波组分别为{haar['train_ilm_opl_pcc']:.4f}/{haar['validation_ilm_opl_pcc']:.4f}，"
              f"完整FlowOCT分别为{full['train_ilm_opl_pcc']:.4f}/{full['validation_ilm_opl_pcc']:.4f}。",
              '在这次固定400次的检查中，无小波组的投影PCC和SSIM均高于另外两组。',
              '无小波组从200到400次仍有改善，而仅小波与完整FlowOCT的训练投影PCC/SSIM都下降；不能把最后一次更新自动当作各组的最佳状态。',
              '仅小波与完整FlowOCT的八步差别较小，移除三项辅助监督没有让仅小波组充分拟合训练血管。',
              '这表明本轮设置下，加入小波和辅助监督没有带来这些投影指标的收益；不是证明它们在完整训练或其他预处理下必然无效。',
              '07与08还有网络内部空间尺度和计算量差异，不能把差距全部归因于小波变换本身。', '',
              '这是单病例、单种子、固定400次更新的模块对照，不能代表充分收敛后的排名。',
              '去掉辅助监督后好转或下降，都需要结合训练曲线和实际血管对应关系解释，不能把纹理变多直接判为血管更真实。', '',
              '[六组原实验及新增两组总表](index.html) · [指标CSV](flowoct_ablation_summary.csv)', '',
              '## 原尺寸对照', '每张图顺序：真实OCTA、仅流匹配、仅小波、完整FlowOCT。', '']
    notes += ['![' + name + '](' + name + ')' for name in figures]
    (reports / 'FLOWOCT_ABLATIONS.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'figures': figures}), flush=True)


if __name__ == '__main__':
    main()
