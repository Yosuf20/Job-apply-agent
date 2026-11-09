from jobagent.matching.skills import normalize, compare_skills
from jobagent.models import Bullet, JobRequirements, Profile, Selection, Experience, Project


"""Scoring bullets points acc to skills they have and in JD 
skills which match with  must have skills have high score and then nice to have have lower score and 
then skills which match keywords of JD"""

WEIGHTS = {"must": 3.0, "nice": 1.0, "keyword": 0.5}
LIMITS = {"max_bullets_per_exp": 4, "max_bullet_per_project": 2, "total_projects": 2}

#whole word matching eg machine learning
#repeated words
#normalise requirements
#add padding to only match the exact words

def score_per_level(skills : list[str], padded_text : str, seen : set, score : float, weight : float):
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

    score_must = score_per_level(req.must_have_skills, padded_text, skills_seen, score, WEIGHTS["must"])
    score_nice = score_per_level(req.nice_to_have_skills, padded_text, skills_seen, score, WEIGHTS["nice"])
    score_key = score_per_level(req.keywords, norm_bullet, padded_text, skills_seen, score, WEIGHTS["keyword"])
         
    return score_must + score_nice + score_key

def select_content(profile : Profile, req : JobRequirements, limits) -> Selection:
    best_projects = []
    for project in profile.projects:
        first_bullet = {}
        sec_bullet = {}
        for bullet in project.bullets:
            temp_score = score_bullet(bullet, req)
            if not first_bullet:
                first_bullet["bullet"] = bullet
                first_bullet["score"] = temp_score
            elif temp_score > first_bullet["score"]: 
                sec_bullet["bullet"] = first_bullet["bullet"]
                sec_bullet["score"] = first_bullet["score"]
                first_bullet["bullet"] = bullet
                first_bullet["score"] = temp_score
            elif not sec_bullet or temp_score > sec_bullet["score"]:
                sec_bullet["bullet"] = bullet
                sec_bullet["score"] = temp_score

        if first_bullet and sec_bullet:    
            best_projects.append(first_bullet)
            best_projects.append(sec_bullet)
    

    for exp in profile.experience:
        bullet_in_exp = []
        for exp_bullet in exp.bullets: 
            temp = score_bullet(exp_bullet, req)
            bullet_in_exp.append((exp_bullet, temp)) 
        bullet_in_exp.sort(key=lambda item: item[1], reverse=True)

        selected_bullets = bullet_in_exp[:LIMITS["max_bullets_per_exp"]]
            
        
         
         
         
