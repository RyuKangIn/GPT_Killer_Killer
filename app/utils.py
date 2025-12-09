import math
import re
from collections import Counter
from typing import Dict

# 자주 쓰는 한국어 논리 연결어들
KOREAN_CONNECTIVES = {
    "또한", "하지만", "그러나", "결론적으로", "요약하자면",
    "게다가", "한편", "따라서", "즉", "때문에", "그럼에도",
}

# 형식적인 문장 종결
FORMAL_ENDINGS = ("니다.", "니다!", "니다?", "다.", "다!", "다?")


def simple_tokenize(text: str):
    """아주 단순한 토크나이저 (실제에선 mecab/khaiii 등으로 교체 추천)"""
    tokens = re.findall(r"[\w]+", text)
    return [t for t in tokens if t.strip()]


def split_sentences(text: str):
    """. ? ! 기준의 단순 문장 분리"""
    raw = re.split(r"(?<=[\.!?])\s+|\n+", text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences

def compute_features(text: str) -> Dict[str, float]:
    sentences = split_sentences(text)
    tokens = simple_tokenize(text)
    n_tokens = len(tokens)
    n_sent = len(sentences) if sentences else 1

    if n_tokens == 0:
        return {
            "length_tokens": 0.0,
            "type_token_ratio": 0.0,
            "avg_sentence_len": 0.0,
            "sentence_burstiness": 0.0,
            "repetition_ratio": 0.0,
            "formal_ending_ratio": 0.0,
            "connectives_per_sentence": 0.0,
        }

    # 기본 통계
    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / n_tokens

    # 문장 길이 분포
    sent_lens = [len(simple_tokenize(s)) for s in sentences]
    avg_sentence_len = sum(sent_lens) / n_sent if n_sent else 0.0
    if len(sent_lens) > 1:
        mean = avg_sentence_len
        var = sum((l - mean) ** 2 for l in sent_lens) / (len(sent_lens) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0

    # Burstiness: std / mean
    if avg_sentence_len > 0:
        burstiness = std / avg_sentence_len
    else:
        burstiness = 0.0

    # 반복도: 가장 많이 나온 토큰 비율
    counts = Counter(tokens)
    most_common_freq = counts.most_common(1)[0][1]
    repetition_ratio = most_common_freq / n_tokens

    # 형식적인 종결어 비율
    formal_count = 0
    for s in sentences:
        for fe in FORMAL_ENDINGS:
            if s.endswith(fe):
                formal_count += 1
                break
    formal_ending_ratio = formal_count / n_sent if n_sent else 0.0

    # 논리 연결어 빈도
    connective_count = sum(1 for t in tokens if t in KOREAN_CONNECTIVES)
    connectives_per_sentence = connective_count / n_sent if n_sent else 0.0

    return {
        "length_tokens": float(n_tokens),
        "type_token_ratio": float(type_token_ratio),
        "avg_sentence_len": float(avg_sentence_len),
        "sentence_burstiness": float(burstiness),
        "repetition_ratio": float(repetition_ratio),
        "formal_ending_ratio": float(formal_ending_ratio),
        "connectives_per_sentence": float(connectives_per_sentence),
    }


def score_ai_likelihood(features: Dict[str, float]) -> Dict[str, float]:
    """
    Heuristic scoring:
    - type_token_ratio 낮을수록 → AI 쪽으로
    - sentence_burstiness 낮을수록 → AI 쪽으로
    - formal_ending_ratio 높을수록 → AI 쪽으로
    - connectives_per_sentence 높을수록 → AI 쪽으로
    - repetition_ratio 높을수록 → AI 쪽으로
    """
    ttr = features["type_token_ratio"]
    burst = features["sentence_burstiness"]
    formal = features["formal_ending_ratio"]
    conn = features["connectives_per_sentence"]
    length = features["length_tokens"]
    repetition = features["repetition_ratio"]

    # 너무 짧은 텍스트는 신뢰도 낮으니 점수 폭을 줄임
    if length < 30:
        base_penalty = 0.1  # 스코어를 0.5 근처로 당기는 용도
    else:
        base_penalty = 0.0

    # TTR: [0.3, 0.8] 구간을  [1, 0]으로 매핑
    ttr_score = 1.0 - max(0.0, min(1.0, (ttr - 0.3) / (0.8 - 0.3)))
    # Burstiness: [0.0, 0.8] -> [1, 0] (너무 균일하면 AI)
    burst_score = 1.0 - max(0.0, min(1.0, burst / 0.8))
    # Formal endings: [0.0, 0.8] -> [0, 1]
    formal_score = max(0.0, min(1.0, formal / 0.8))
    # Connectives: [0.0, 0.8] -> [0, 1]
    conn_score = max(0.0, min(1.0, conn / 0.8))
    # Repetition: [0.0, 0.2] -> [0, 1]
    rep_score = max(0.0, min(1.0, repetition / 0.2))

    # 가중 합산
    ai_score_raw = (
        0.30 * ttr_score +
        0.20 * burst_score +
        0.20 * formal_score +
        0.15 * conn_score +
        0.15 * rep_score
    )

    ai_score = max(0.0, min(1.0, ai_score_raw + base_penalty))

    feature_scores = {
        "ttr_score": ttr_score,
        "burstiness_score": burst_score,
        "formal_score": formal_score,
        "connectives_score": conn_score,
        "repetition_score": rep_score,
    }

    feature_scores["ai_score_raw"] = ai_score_raw
    feature_scores["ai_score"] = ai_score

    return feature_scores


def label_from_score(score: float) -> str:
    if score >= 0.7:
        return "AI_LIKELY"
    elif score >= 0.4:
        return "UNCERTAIN"
    else:
        return "HUMAN_LIKELY"


