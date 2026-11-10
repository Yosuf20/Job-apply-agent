from jobagent.matching.skills import normalize, compare_skills
from jobagent.models import Bullet, JobRequirements, Profile, Selection, Experience, Project
from pydantic import BaseModel

class SelectionLimits(BaseModel):
    max_bullets_per_experience: int
    max_bullets_per_project: int
    max_projects: int
    max_total_bullets: int


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
    score_key = score_per_level(req.keywords, padded_text, skills_seen, score, WEIGHTS["keyword"])
         
    return score_must + score_nice + score_key

def select_content(profile : Profile, req : JobRequirements, limits : SelectionLimits) -> Selection:
    best_projects = []
    for project in profile.projects:
        curr_project = []
        for bullet in project.bullets:
            temp_score = score_bullet(bullet, req)
            curr_project.append((bullet, temp_score))

        ranked_bullet = sorted(curr_project, key=lambda item: item[1], reverse=True)
        selected = ranked_bullet[:limits.max_bullets_per_project]

        tech_stack_score = 0
        for tech in project.tech:
            if (tech in req.must_have_skills or tech in req.nice_to_have_skills or tech in req.keywords):
                tech_stack_score += 2
        pro_score = sum(item[1] for item in curr_project)

        best_projects.append(
            (selected, pro_score + tech_stack_score)
        )

    ranked_projects = sorted(best_projects, key=lambda item: item[1], reverse=True)
    selected_projects = ranked_projects[:limits.max_projects]
    
    selected_experience = []
    for exp in profile.experience:
        bullet_in_exp = []
        for exp_bullet in exp.bullets: 
            temp = score_bullet(exp_bullet, req)
            bullet_in_exp.append((exp_bullet, temp)) 

        bullet_in_exp.sort(key=lambda item: item[1], reverse=True)
        selected_bullets = bullet_in_exp[:limits.max_bullets_per_experience]

        selected_experience.append(exp.model_copy(update={"bullet": selected_bullets}))
        


    return Selection(selected_experience=selected_experience, selected_projects=selected_projects)

            
        
         
         
         
