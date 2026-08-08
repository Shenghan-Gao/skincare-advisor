"""L1 -- ethics & safety. TEAMMATE C owns this; works against mocks from day 1."""
from app.schemas import AdvisorResponse, UserProfile
from skincare.rag.retrieve import load_rules

DISCLAIMER = (
    "This tool offers cosmetic product suggestions only. It is not medical advice "
    "and cannot diagnose any skin condition. For persistent, painful, or worsening "
    "skin problems, please consult a licensed dermatologist."
)


def apply_safety(resp: AdvisorResponse, profile: UserProfile) -> AdvisorResponse:
    rules = load_rules()
    flags: list[str] = list(resp.safety_flags)   # 保留生成器已有的 flag,不要覆盖
    kept = []
    for rec in resp.recommendations:
        ings = " ".join(rec.key_ingredients).lower()
        if profile.pregnant and any(u in ings for u in rules["pregnancy_unsafe"]):
            flags.append(f"removed {rec.name}: pregnancy-unsafe ingredient")
            continue
        avoid = [a.lower() for a in profile.avoid_ingredients if a.strip()]  # 空串会匹配一切
        if any(a in ings for a in avoid):
            flags.append(f"removed {rec.name}: user-avoided ingredient")
            continue
        kept.append(rec)

    resp.recommendations = kept
    resp.safety_flags = flags
    resp.disclaimer = DISCLAIMER
    return resp
