import numpy as np
import os
from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
    get_natural_goal_conditions,
    get_object_scope,
)
# from bddl.parsing import scan_tokens, parse_predicates, parse_action, parse_problem
# from bddl.config import SUPPORTED_BDDL_REQUIREMENTS as supported_requirements
# from bddl_debug_backend import DebugBackend, DebugGenericObject

def verify_bddl(activity):
    # for example activity == "cleaning_up_after_a_meal"
    conds = Conditions(activity, 0, "omnigibson")
    scope = get_object_scope(conds)
    # init_conds = get_initial_conditions(conds, OmniGibsonBDDLBackend(), scope, generate_ground_options=False)
    # goal_conds = get_goal_conditions(conds, OmniGibsonBDDLBackend(), scope, generate_ground_options=False)
    # print(get_natural_goal_conditions(conds)) # test
    # ground_goal_state_options = get_ground_goal_state_options(conds, OmniGibsonBDDLBackend(), scope, goal_conds)
    judge, satisfied_predicates = evaluate_goal_conditions(goal_conds)
    return judge
