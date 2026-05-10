"""
stage_b.py — Stage B: 우선순위 가이드라인 생성
  취약도(severity) × 중재 가능성(modifiability) → 중재 우선순위
"""

import numpy as np
from config import MODIFIABILITY


def compute_priority(domain_results: dict) -> list:
    """
    각 영역의 Z-score(severity)와 문헌 기반 가중치(modifiability)를 결합하여
    중재 우선순위를 산출.
    
    Priority Score = |Z-score| × modifiability_weight
    → 높을수록 "급하면서 개선 가능성도 있는" 영역
    """
    priorities = []

    for domain_name, info in domain_results.items():
        z = info['z_score']
        severity = abs(z)  # Z-score 절댓값 = 얼마나 나쁜가

        mod_info = MODIFIABILITY[domain_name]
        mod_weight = mod_info['weight']

        # Priority Score = severity × modifiability
        priority_score = severity * mod_weight

        # 정상 범위(Z > -1)이면 중재 불필요 → 점수 0
        if z > -1.0:
            priority_score = 0.0
            needs_intervention = False
        else:
            needs_intervention = True

        priorities.append({
            'domain': domain_name,
            'z_score': z,
            'severity': severity,
            'severity_label': info['severity'],
            'modifiability': mod_weight,
            'mod_rationale': mod_info['rationale'],
            'priority_score': priority_score,
            'needs_intervention': needs_intervention,
            'interventions': mod_info['interventions'],
        })

    # Priority Score 내림차순 정렬
    priorities.sort(key=lambda x: x['priority_score'], reverse=True)

    # 순위 부여 (중재 불필요 영역 제외)
    rank = 1
    for p in priorities:
        if p['needs_intervention']:
            p['rank'] = rank
            rank += 1
        else:
            p['rank'] = None

    return priorities


def generate_guideline_text(classification: dict, priorities: list) -> str:
    """
    의사에게 제공할 가이드라인 텍스트 생성.
    """
    subtype = classification['subtype']

    # 중재 필요 영역만 필터
    actionable = [p for p in priorities if p['needs_intervention']]

    if not actionable:
        return f"""
## 중재 가이드라인

Subtype: 비위축 AD — {subtype}

현재 모든 인지 영역이 CN 정상 범위(Z > -1.0) 내에 있어, 특정 영역에 대한 집중 중재보다는 
전반적 인지 유지 프로그램 (인지 자극 치료, 규칙적 운동, 사회적 활동)을 권고합니다.
"""

    # 가이드라인 생성
    text = f"""
## 중재 가이드라인

Subtype: 비위축 AD — {subtype}

이 환자는 비위축 AD {subtype}으로, 해마 위축이 없음에도 아래 영역에서 인지 저하가 확인됩니다.
중재 우선순위는 취약도(CN 대비 Z-score)와 중재 가능성(문헌 근거)을 함께 고려하여 산출했습니다.

---

### 중재 우선순위

"""

    for p in actionable:
        text += f"""우선순위 {p['rank']}. {p['domain']} (Priority Score: {p['priority_score']:.2f})
- 취약도: Z = {p['z_score']:.1f} ({p['severity_label']})
- 중재 가능성: {p['modifiability']:.0%} — {p['mod_rationale']}
- 권고 중재:
"""
        for intervention in p['interventions']:
            text += f"  - {intervention}\n"
        text += "\n"

    # 요약 문장
    if len(actionable) >= 2:
        top1 = actionable[0]
        top2 = actionable[1]
        text += f"""---

### 요약

> 이 환자는 비위축 AD {subtype}으로, {top1['domain']}(Z={top1['z_score']:.1f})과 
> {top2['domain']}(Z={top2['z_score']:.1f})이 주요 취약 영역입니다. 
> 문헌 근거에 기반하여 {top1['interventions'][0]}(우선순위 1)과 
> {top2['interventions'][0]}(우선순위 2)을 고려할 수 있습니다.

본 가이드라인은 ADNI 기반 비위축 AD subtype 연구 결과에 근거한 참고 자료이며, 
최종 치료 결정은 담당 의사의 임상적 판단에 따릅니다.
"""
    elif len(actionable) == 1:
        top1 = actionable[0]
        text += f"""---

### 요약

> 이 환자는 비위축 AD {subtype}으로, {top1['domain']}(Z={top1['z_score']:.1f})이 
> 주요 취약 영역입니다. {top1['interventions'][0]}을 우선적으로 고려할 수 있습니다.

본 가이드라인은 참고 자료이며, 최종 결정은 담당 의사의 판단에 따릅니다.
"""

    return text


def generate_priority_table(priorities: list) -> list:
    """Streamlit 테이블용 데이터."""
    table = []
    for p in priorities:
        table.append({
            '순위': f"#{p['rank']}" if p['rank'] else '-',
            '영역': p['domain'],
            'Z-score': f"{p['z_score']:.1f}",
            '심각도': p['severity_label'],
            '중재 가능성': f"{p['modifiability']:.0%}",
            'Priority Score': f"{p['priority_score']:.2f}",
            '중재 필요': '✅' if p['needs_intervention'] else '—',
        })
    return table
