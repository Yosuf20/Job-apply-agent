from jobagent.matching.skills import normalize, compare_skills
from jobagent.models import Bullet, JobRequirements, Profile
import re
"""Scoring bullets points acc to skills they have and in JD 
skills which match with  must have skills have high score and then nice to have have lower score and 
then skills which match keywords of JD"""

WEIGHTS = {"must": 3.0, "nice": 1.0, "keyword": 0.5}

#whole word matching eg machine learning
#repeated words
#normalise requirements

def score_per_level(skills : str, padded_text : str, seen : set, score : float, weight : float):
    for skill in skills:
        key = normalize(skill)
        if not key or key in seen:
                continue
        elif f" {key} " in padded_text:
                score += weight
                seen.add(key)
    return score

def score_bullet(bullet : Bullet, req : JobRequirements) -> float:
    skills_seen = set()
    score = 0
    norm_bullet = normalize(f"{bullet.text} {" ".join(bullet.tags)}")
    padded_text = f" {norm_bullet} "
    #norm_must = normalize(" ".join(req.must_have_skills))
    #norm_nice = normalize(" ".join(req.nice_to_have_skills))
    #norm_key = normalize(" ".join(req.keywords))

    score_must = score_per_level(req.must_have_skills, padded_text, skills_seen, score, WEIGHTS["must"])
    score_nice = score_per_level(req.nice_to_have_skills, padded_text, skills_seen, score, WEIGHTS["nice"])
    score_key = score_per_level(req.keywords, norm_bullet, padded_text, skills_seen, score, WEIGHTS["keyword"])
         
    return score_must + score_nice + score_key

def select_content(profile : Profile, req : JobRequirements, limits):
    pass
