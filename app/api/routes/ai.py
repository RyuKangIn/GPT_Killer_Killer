from fastapi import APIRouter
from app.models import GPTKillerResponse, GPTKillerRequest
from app.utils import compute_features, score_ai_likelihood, label_from_score

router = APIRouter(prefix="/ai", tags=["ai"])

@router.post("/gpt_killer", response_model=GPTKillerResponse)
async def gpt_killer(request: GPTKillerRequest) -> GPTKillerResponse:
    text = request.text
    features = compute_features(text)
    feature_scores = score_ai_likelihood(features)
    ai_score = feature_scores["ai_score"]
    label = label_from_score(ai_score)

    meta = {
        "length_tokens": features["length_tokens"],
        "avg_sentence_len": features["avg_sentence_len"],
        "sentence_burstiness": features["sentence_burstiness"],
        "type_token_ratio": features["type_token_ratio"],
        "formal_ending_ratio": features["formal_ending_ratio"],
        "connectives_per_sentence": features["connectives_per_sentence"],
        "repetition_ratio": features["repetition_ratio"],
    }

    return GPTKillerResponse(
        ai_score=ai_score,
        label=label,
        feature_scores=feature_scores,
        meta=meta,
    )