from jobagent.matching.skills import normalize, compare_skills
from jobagent.models import Bullet, JobRequirements, Profile, Selection, SelectedProject, SelectionLimits, SelectedExperience, SelectedBullet, Project
from pydantic import BaseModel



"""Scoring bullets points acc to skills they have and in JD 
skills which match with  must have skills have high score and then nice to have have lower score and 
then skills which match keywords of JD"""

WEIGHTS = {"must": 3.0, "nice": 1.0, "keyword": 0.5}
LIMITS = {"max_bullets_per_exp": 4, "max_bullet_per_project": 2, "total_projects": 2}

#whole word matching eg machine learning
#repeated words
#normalise requirements
#add padding to only match the exact words

PROJECT_TOP_BULLET = 2
PROJECT_TECH_BONUS = 2

Selected = tuple[int, Bullet, float]

def _score_per_level(skills : list[str], padded_text : str, seen : set, score : float, weight : float):
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

    score_must = _score_per_level(req.must_have_skills, padded_text, skills_seen, score, WEIGHTS["must"])
    score_nice = _score_per_level(req.nice_to_have_skills, padded_text, skills_seen, score, WEIGHTS["nice"])
    score_key = _score_per_level(req.keywords, padded_text, skills_seen, score, WEIGHTS["keyword"])
         
    return score_must + score_nice + score_key


def _score_entry(bullets: list[Bullet], req: JobRequirements) ->list[Selected]:
    """Score every bullet of one job or project, remembering its position."""
    return [(pos, b, score_bullet(b, req)) for pos, b in enumerate(bullets)]

def _to_selected(scored: list[Selected], position: set[int]) -> list[SelectedBullet]:
    """it only created SelectedBullet object if the index is in the positon(parameter)"""
    return [
        SelectedBullet(
            source_id=bullet.id,
            text=bullet.text,
            score = score,
        )
        for pos, bullet, score in scored
        if pos in position 
    ]

def _pick_project_positons(scored: list[Selected], max_n: int):
    """top 3 best scoring bullets. if all bullets scores 0 then the first one"""
    rank = []
    if not scored:
        return set()
    rank = sorted(scored, key=lambda item: (-item[2], item[0]))
    picked = {rank[0][0]}
    for idx, _, sc in rank:
        if len(picked) >= max_n:
            break
        if sc > 0:
            picked.add(idx)

    return picked

def _project_score(scored, project: Project, req):
    score = sorted((s for _, _, s in scored), reverse=True)[:PROJECT_TOP_BULLET]
    wanted = set()
    for skill in [*req.must_have_skills, *req.nice_to_have_skills, *req.keywords]:
        wanted.add(normalize(skill))
    tech = {normalize(t) for t in project.tech}

    bonus_count = 0
    for t in tech:
        if t in wanted:
            bonus_count += 1
    return bonus_count*PROJECT_TECH_BONUS + sum(score)


def _select_projects(profile: Profile, req: JobRequirements, limits: SelectionLimits) -> list[SelectedProject]:
    ranked = []
    for idx, p in enumerate(profile.projects):
        b_scored = _score_entry(p.bullets, req)
        ranked.append((_project_score(b_scored, p, req), idx, p, b_scored))

    ranked.sort(key=lambda item: item[0], reverse=True)   
    key = ranked[:limits.max_projects]     
    key.sort(key=lambda item:item[1])

    return [
        SelectedProject(
            name=project.name,
            bullets=(_pick_project_positons(score, max_n=limits.max_bullets_per_project)),
            score=score,
            tech=list(project.tech
            ),
        )
        for score, _, project, bull in key
    ]



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
    ranked_projects = ranked_projects[:limits.max_projects]
    final_pro = [item[0] for item in ranked_projects]
    selected_projects = final_pro
    
    selected_experience = []
    for exp in profile.experience:
        bullet_in_exp = []
        for exp_bullet in exp.bullets: 
            temp = score_bullet(exp_bullet, req)
            bullet_in_exp.append((exp_bullet, temp)) 

        bullet_in_exp.sort(key=lambda item: item[1], reverse=True)
        selected_bullets = bullet_in_exp[:limits.max_bullets_per_experience]
        selected_bullets = [item[0] for item in selected_bullets]
        selected_experience.append(exp.model_copy(update={"bullets": selected_bullets}))
        


    return Selection(selected_experience=selected_experience, selected_projects=selected_projects)

            
        
         
         
         
