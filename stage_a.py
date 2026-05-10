"""
stage_a.py — Stage A: 환자 프로파일링
  1. 6개 인지검사 → K-Means subtype 분류
  2. 5개 영역별 CN 대비 Z-score 산출
  3. 레이더 차트 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

from config import CN_REFERENCE, CLUSTER_CENTERS_RAW, SCALER_PARAMS, DOMAINS, COLORS


def classify_subtype(scores: dict) -> dict:
    """
    6개 인지검사 점수를 입력받아 subtype 분류.
    scores: {'CATANIMSC': 10, 'TRABSCOR': 200, 'LDELTOTL': 1, ...}
    """
    # 표준화 (비위축 AD 51명 기준 scaler)
    scaled = {}
    for var in SCALER_PARAMS:
        scaled[var] = (scores[var] - SCALER_PARAMS[var]['mean']) / SCALER_PARAMS[var]['std']

    # 각 클러스터 중심까지 유클리드 거리
    distances = {}
    for c_id, center in CLUSTER_CENTERS_RAW.items():
        center_scaled = {}
        for var in SCALER_PARAMS:
            center_scaled[var] = (center[var] - SCALER_PARAMS[var]['mean']) / SCALER_PARAMS[var]['std']
        dist = np.sqrt(sum((scaled[v] - center_scaled[v]) ** 2 for v in SCALER_PARAMS))
        distances[c_id] = dist

    cluster = min(distances, key=distances.get)
    confidence = 1.0 - (min(distances.values()) / sum(distances.values()))

    subtype = '인지보존형' if cluster == 0 else '인지저하형'

    return {
        'cluster': cluster,
        'subtype': subtype,
        'distances': distances,
        'confidence': confidence,
    }


def compute_domain_zscores(scores: dict) -> dict:
    """
    5개 영역별 CN 대비 Z-score 산출.
    Z-score가 음수 = CN보다 나쁨 (모든 영역을 "높을수록 좋음" 방향으로 통일)
    """
    domain_results = {}

    for domain_name, domain_info in DOMAINS.items():
        z_scores = []
        var_details = []

        for var in domain_info['variables']:
            ref = CN_REFERENCE[var]
            raw = scores[var]

            if ref['direction'] == 'higher_better':
                z = (raw - ref['mean']) / ref['std']
            else:
                # 낮을수록 좋은 변수는 부호 반전 (높은 TRABSCOR = 나쁨 → 음수 Z)
                z = -(raw - ref['mean']) / ref['std']

            # Z-score cap: -4 ~ +2 (CN 기준에서 극단적인 값 방지)
            z = max(-4.0, min(2.0, z))

            z_scores.append(z)
            var_details.append({
                'variable': var,
                'label': ref['label'],
                'raw_score': raw,
                'cn_mean': ref['mean'],
                'cn_std': ref['std'],
                'z_score': z,
            })

        avg_z = np.mean(z_scores)

        # 심각도 판정
        if avg_z >= -1.0:
            severity = '정상 범위'
            severity_color = '#2ECC71'
        elif avg_z >= -2.0:
            severity = '경도 저하'
            severity_color = '#F39C12'
        elif avg_z >= -3.0:
            severity = '중등도 저하'
            severity_color = '#E67E22'
        else:
            severity = '심도 저하'
            severity_color = '#E74C3C'

        domain_results[domain_name] = {
            'z_score': avg_z,
            'severity': severity,
            'severity_color': severity_color,
            'variables': var_details,
            'description': domain_info['description'],
            'brain_region': domain_info['brain_region'],
        }

    return domain_results


def plot_radar_chart(domain_results: dict, patient_id: str = 'Patient') -> plt.Figure:
    """
    5개 영역 Z-score 레이더 차트.
    0 = CN 평균, 안쪽(음수) = CN보다 나쁨.
    """
    domains = list(domain_results.keys())
    z_values = [domain_results[d]['z_score'] for d in domains]

    # 레이더 차트 각도
    N = len(domains)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    z_values_plot = z_values + [z_values[0]]

    # CN 기준선 (Z=0)
    cn_baseline = [0] * (N + 1)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    # CN 정상 범위 (-1 ~ 0)
    cn_minus1 = [-1] * (N + 1)
    ax.fill(angles, cn_baseline, alpha=0.1, color='green', label='CN Mean (Z=0)')
    ax.plot(angles, cn_baseline, color='green', linewidth=2, linestyle='--')
    ax.plot(angles, cn_minus1, color='orange', linewidth=1, linestyle=':', alpha=0.5)

    # 환자 프로파일
    color = COLORS['impaired'] if any(z < -1.5 for z in z_values) else COLORS['preserved']
    ax.fill(angles, z_values_plot, alpha=0.25, color=color)
    ax.plot(angles, z_values_plot, color=color, linewidth=2.5, marker='o', markersize=8, label=patient_id)

    # 각 꼭짓점에 Z-score 표시
    for i, (angle, z) in enumerate(zip(angles[:-1], z_values)):
        severity_color = domain_results[domains[i]]['severity_color']
        ax.annotate(f'Z={z:.1f}', xy=(angle, z), fontsize=10, fontweight='bold',
                    ha='center', va='bottom', color=severity_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=severity_color, alpha=0.9))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=11, fontweight='bold')
    ax.set_ylim(-4, 1)
    ax.set_yticks([-3, -2, -1, 0])
    ax.set_yticklabels(['-3 SD', '-2 SD', '-1 SD', 'CN Mean'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title(f'{patient_id} — Cognitive Domain Profile\n(vs. CN Reference)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def generate_profile_summary(classification: dict, domain_results: dict) -> str:
    """텍스트 요약 생성."""
    subtype = classification['subtype']
    confidence = classification['confidence']

    # 가장 취약한 영역
    sorted_domains = sorted(domain_results.items(), key=lambda x: x[1]['z_score'])
    worst = sorted_domains[0]
    second = sorted_domains[1]

    summary = f"""
## 환자 프로파일 요약

**Subtype:** {subtype} (confidence: {confidence:.0%})

**가장 취약한 영역:**
1. {worst[0]} (Z = {worst[1]['z_score']:.1f}, {worst[1]['severity']}) — {worst[1]['description']}
2. {second[0]} (Z = {second[1]['z_score']:.1f}, {second[1]['severity']}) — {second[1]['description']}

**영역별 상세:**
"""
    for domain, info in sorted_domains:
        summary += f"- **{domain}** (Z={info['z_score']:.1f}): {info['severity']} | 관련 뇌 영역: {info['brain_region']}\n"
        for var_detail in info['variables']:
            summary += f"  - {var_detail['label']}: {var_detail['raw_score']} (CN 평균 {var_detail['cn_mean']:.1f} ± {var_detail['cn_std']:.1f})\n"

    return summary
