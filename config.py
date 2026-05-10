"""
config.py — CDSS 설정값
  CN 기준 통계량, 5개 인지 영역 정의, 중재 가능성 가중치
  
  ⚠️ CN_REFERENCE 값은 ADNI CN군의 실제 평균/SD로 교체해야 함.
     현재 값은 ADNI 문헌 기반 추정치.
"""

# ============================================================
# CN 기준 통계량 (정상인 평균/SD)
# → 실제 ADNI CN 데이터에서 산출한 값으로 교체할 것
# ============================================================
CN_REFERENCE = {
    'CATANIMSC': {'mean': 20.9, 'std': 5.3,  'direction': 'higher_better', 'label': '언어유창성 (동물 이름대기)'},
    'TRABSCOR':  {'mean': 83.6, 'std': 43.2, 'direction': 'lower_better',  'label': '실행기능 (Trail Making B, 초)'},
    'LDELTOTL':  {'mean': 13.3,  'std': 3.1,  'direction': 'higher_better', 'label': '기억력 (지연회상)'},
    'MMSCORE':   {'mean': 29.0, 'std': 1.2,  'direction': 'higher_better', 'label': 'MMSE (전반적 인지)'},
    'CDRSB':     {'mean': 0.05, 'std': 0.16, 'direction': 'lower_better',  'label': '치매 중증도 (CDR-SB)'},
    'ADASTT13':  {'mean': 9.0,  'std': 4.4,  'direction': 'lower_better',  'label': 'ADAS-Cog 13 (종합 인지평가)'},
}

# ============================================================
# K-Means 클러스터 중심 (원점수, CN 5th %ile 기준)
# → 실제 학습된 모델의 cluster_centers_로 교체할 것
# ============================================================
CLUSTER_CENTERS_RAW = {
    0: {'CATANIMSC': 15.5, 'TRABSCOR': 133.3, 'LDELTOTL': 2.7, 'MMSCORE': 25.1, 'CDRSB': 3.7, 'ADASTT13': 23.3},  # 인지보존형
    1: {'CATANIMSC': 9.5,  'TRABSCOR': 236.5, 'LDELTOTL': 0.8, 'MMSCORE': 22.6, 'CDRSB': 5.0, 'ADASTT13': 33.9},  # 인지저하형
}

# 표준화에 사용한 scaler 파라미터 (비위축 AD 51명 기준)
SCALER_PARAMS = {
    'CATANIMSC': {'mean': 11.7, 'std': 5.4},
    'TRABSCOR':  {'mean': 198.1, 'std': 84.1},
    'LDELTOTL':  {'mean': 1.6, 'std': 2.0},
    'MMSCORE':   {'mean': 23.5, 'std': 2.2},
    'CDRSB':     {'mean': 4.5, 'std': 1.6},
    'ADASTT13':  {'mean': 30.0, 'std': 8.4},
}

# ============================================================
# 5개 인지 영역 정의
# ============================================================
DOMAINS = {
    'Memory': {
        'variables': ['LDELTOTL'],
        'description': 'Hippocampal-dependent episodic memory (Delayed Recall)',
        'brain_region': 'Medial Temporal Lobe / Hippocampus',
    },
    'Executive Function': {
        'variables': ['TRABSCOR'],
        'description': 'Planning, shifting, attention (Trail Making B)',
        'brain_region': 'Frontal Lobe',
    },
    'Language/Semantic': {
        'variables': ['CATANIMSC'],
        'description': 'Semantic network access (Category Fluency - Animals)',
        'brain_region': 'Temporal Lobe',
    },
    'Global Cognition': {
        'variables': ['MMSCORE', 'ADASTT13'],
        'description': 'Composite cognitive measures (MMSE + ADAS-Cog)',
        'brain_region': 'Multi-region',
    },
    'Daily Function': {
        'variables': ['CDRSB'],
        'description': 'Functional impairment (CDR Sum of Boxes)',
        'brain_region': 'Global',
    },
}

# ============================================================
# 중재 가능성 가중치 (문헌 기반)
# 0.0 = 효과 없음, 1.0 = 강한 효과 근거
# ============================================================
MODIFIABILITY = {
    'Memory': {
        'weight': 0.5,
        'rationale': '에피소드 기억 훈련의 효과는 제한적이나, 보상 전략(외부 기억 보조, 간격 회상)은 일부 효과 보고',
        'interventions': ['보상 전략 기반 기억 훈련 (간격 회상법)', '외부 기억 보조 도구 (알림, 메모)'],
    },
    'Executive Function': {
        'weight': 0.8,
        'rationale': '실행기능 훈련(목표 관리 훈련, 인지 자극 프로그램)은 MCI 및 경도 AD에서 비교적 강한 효과 근거',
        'interventions': ['목표 관리 훈련 (Goal Management Training)', '전산화 인지 훈련 (실행기능 모듈)', '이중과제 훈련'],
    },
    'Language/Semantic': {
        'weight': 0.6,
        'rationale': '의미 유창성 훈련, 명명 훈련이 중등도 효과. 의미 네트워크 강화 접근',
        'interventions': ['의미 유창성 훈련 (범주 생성)', '명명 훈련 (그림 이름대기)', '의미 연관 훈련'],
    },
    'Global Cognition': {
        'weight': 0.7,
        'rationale': '다영역 인지 자극 프로그램(CST)이 경도-중등도 AD에서 MMSE, ADAS-Cog 개선 근거',
        'interventions': ['인지 자극 치료 (Cognitive Stimulation Therapy)', '다영역 인지 훈련 프로그램', '음악 치료 + 인지 자극 병합'],
    },
    'Daily Function': {
        'weight': 0.65,
        'rationale': '작업치료 기반 일상생활 훈련, 보호자 교육이 CDR-SB 악화 지연에 효과',
        'interventions': ['작업치료 기반 일상생활 훈련', '보호자 교육 프로그램', '환경 수정 및 보상 전략'],
    },
}

# ============================================================
# 시각화 설정
# ============================================================
SEVERITY_THRESHOLDS = {
    'normal': (-1.0, '정상 범위'),
    'mild': (-2.0, '경도 저하'),
    'moderate': (-3.0, '중등도 저하'),
    'severe': (float('-inf'), '심도 저하'),
}

COLORS = {
    'preserved': '#2ECC71',
    'impaired': '#E74C3C',
    'primary': '#3498DB',
    'warning': '#F39C12',
    'bg': '#F8F9FA',
}
