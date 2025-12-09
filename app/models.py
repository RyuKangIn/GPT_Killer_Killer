from pydantic import field_validator
from sqlmodel import SQLModel, Field
from typing import Dict, Any

class GPTKillerRequest(SQLModel):
    text: str = Field(...)

    @field_validator("text")
    def text_validate(cls, v: str) -> str:
        # 앞뒤 공백 제거
        v = v.strip()

        # 1) 길이 체크 (300자 이내면 너무 짧다)
        if len(v) <= 300:
            raise ValueError("내용이 너무 짧습니다. 최소 301자 이상이어야 합니다.")

        # 2) 공백 제외 문자열 만들기
        stripped = "".join(ch for ch in v if not ch.isspace())
        if not stripped:
            raise ValueError("유효한 문자가 없습니다.")

        # 3) 한국어 비율 검사 (80% 이상이어야 통과)
        korean_count = sum(1 for ch in stripped if "\uac00" <= ch <= "\ud7a3")
        korean_ratio = korean_count / len(stripped)

        if korean_ratio < 0.8:
            raise ValueError("한국어만 검사 가능합니다.")

        return v

class GPTKillerResponse(SQLModel):
    ai_score: float
    label: str
    feature_scores: Dict[str, float]
    meta: Dict[str, Any]
