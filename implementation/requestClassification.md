Initial Prompt from user
loadRoleList() -> roleList
multi-agent understanding | "You are a ..., given your role and understanding of ..., and the classes of requests [...], which class fits best with this request?"
if votedClasses >1 -> requestClass
multi-agent voting | "You are a ..., given this request ..., and you are asked to vote if this request should be understood as [votedClasses], only answer with the one class that fits best"
initClassFlow(requestClass)
