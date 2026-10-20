"""Promotion gate for this small RL case; never changes training or its rewards."""
import argparse
import hashlib
import json
from pathlib import Path
from jaw_env import ROOT


def read(path):
    return json.loads(path.read_text())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["initial","trained","scripted","export_check","learning_check","browser","actor","fine_step","nominal","output"]:
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--negative-open",type=Path,required=True)
    p.add_argument("--negative-closed",type=Path,required=True)
    args=p.parse_args()
    initial,trained,scripted=map(read,[args.initial,args.trained,args.scripted])
    exports,learning,browser=map(read,[args.export_check,args.learning_check,args.browser])
    failures=[]
    def check(ok,label):
        if not ok: failures.append(label)
    check(initial["trials"]==trained["trials"]==scripted["trials"]==12,"12 paired cases required")
    check([r["config"] for r in initial["results"]]==[r["config"] for r in trained["results"]]==[r["config"] for r in scripted["results"]],"case pairing mismatch")
    check(trained["successes"]>=10,"fewer than 10/12 held-out successes")
    check(trained["successes"]-initial["successes"]>=6,"less than 6/12 improvement")
    check(all(r["modelAudit"]["timestep"]==.0000625 for group in [initial,trained,scripted] for r in group["results"]),"paired policies used different execution fidelity")
    for row in trained["results"]:
        s=row["summary"]
        audit=row["modelAudit"]
        check(audit["freePayload"] and audit["freeBase"] and audit["equalities"]==0 and audit["mocapBodies"]==0,
              "held-out attachment or fixed body: "+row["config"]["label"])
        check(s["maxExternal"]==0 and s["finite"] and not s["nonFootSupport"] and
              s["maxPenetration"]<.0005 and s["maxRobotPenetration"]<.0005 and s["maxJointLimitExcess"]<=.002,
              "held-out physical violation: "+row["config"]["label"])
    check(exports["accepted"] and exports["actionMismatches"]==0,"actor export mismatch")
    check(learning["accepted"] and learning["finalSha256"]==trained["modelSha256"],"no matching learned checkpoint")
    check(learning["initialSha256"]==initial["modelSha256"],"initial checkpoint mismatch")
    check(browser["acceptance"]["accepted"],"browser did not pass")
    check(browser["provenance"]["controlMode"]=="learned","browser did not use learned jaw")
    actor_hash=hashlib.sha256(args.actor.read_bytes()).hexdigest()
    check(exports["checkpointSha256"]==trained["modelSha256"] and exports["actorSha256"]==actor_hash,"export check used a different policy")
    check(browser["provenance"]["jawPolicySha256"]==actor_hash,"browser actor hash mismatch")
    audit=browser["modelAudit"]
    check(audit["timestep"]==.0000625 and audit["freePayload"] and audit["freeBase"] and
          audit["equalities"]==0 and audit["mocapBodies"]==0,"browser model/fidelity mismatch")
    check(read(args.negative_open)["successes"]==0 and read(args.negative_closed)["successes"]==0,"constant-action negative control passed")
    fine,nominal=map(read,[args.fine_step,args.nominal])
    check(fine["successes"]==nominal["successes"]==1,"step-size convergence trial failed")
    delta=abs(fine["results"][0]["acceptance"]["minHeldLift"]-nominal["results"][0]["acceptance"]["minHeldLift"])
    check(delta<.001,"lift changed by at least 1 mm at half timestep")
    check(fine["modelSha256"]==nominal["modelSha256"]==trained["modelSha256"],"convergence used a different policy")
    check(fine["results"][0]["modelAudit"]["timestep"]==.00003125 and
          nominal["results"][0]["modelAudit"]["timestep"]==.0000625,"convergence timestep mismatch")
    successes=[r for r in trained["results"] if r["acceptance"]["accepted"]]
    result=dict(accepted=not failures,failures=failures,scope="learned jaw; frozen balance and scripted head motion",
                initialSuccesses=initial["successes"],trainedSuccesses=trained["successes"],
                scriptedSuccesses=scripted["successes"],trials=12,
                minimumHeldLiftMm=min((r["acceptance"]["minHeldLift"]*1000 for r in successes),default=0),
                maximumPenetrationMm=max(r["summary"]["maxPenetration"]*1000 for r in trained["results"]),
                maximumHoldDriftMm=max((r["acceptance"]["holdDrift"]*1000 for r in successes),default=0),
                halfStepLiftDifferenceMm=delta*1000,actorSha256=actor_hash,
                checkpointSha256=trained["modelSha256"],learning=learning,exports=exports,
                browserAcceptance=browser["acceptance"],
                evidence={k:str(v) for k,v in vars(args).items()})
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    assert not failures,failures
