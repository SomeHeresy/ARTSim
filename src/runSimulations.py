import time
from src import globalVar
from src import nub_ctm as ctm


def runSteadyState(logsFile):

    if not globalVar.modelPrepared:
        raise Exception(
            "Model not prepared. Please call prepareModel(model) before running the simulation")

    nodes = globalVar.nodes
    model = globalVar.model

    print("Solving steady state...")
    stepStart = time.perf_counter()

    ssTempVector = ctm.solve_steady_state()

    ssTempVector = [x + globalVar.baseTemp for x in ssTempVector]

    stepEnd = time.perf_counter()
    print(f"Step time: {(stepEnd - stepStart) * 1000:.3f} ms")
    ctm.printInfo(ssTempVector, nodes, model, logsFile)


# NOTE: make sure to call prepareModel before running the simulation
def runTransient(stepDefinition, logsFile):

    if not globalVar.modelPrepared:
        raise Exception(
            "Model not prepared. Please call prepareModel(model) before running the simulation")

    nodes = globalVar.nodes
    model = globalVar.model
    GMatrix = globalVar.GMatrix
    CVector = globalVar.CVector

    steps = len(stepDefinition)
    globalVar.IVectorVector = ctm.populate_I_vector_vector_transient(
        nodes, model, steps)

    # Make initTempVector
    initTempVector = []
    for i in range(len(nodes)):
        initTempVector.append(0)
    # NOTE: The above should NOT start at the ambient temperature. It should start at the initial temperature, where ambient is 0.

    # changed (CY)
    ttemp = ctm.doBeuler(
        GMatrix, CVector, globalVar.IVectorVector, initTempVector, stepDefinition)

    for i in range(len(ttemp)):
        ttemp[i] = [x + globalVar.baseTemp for x in ttemp[i]]

    ctm.saveTransientInfo(ttemp, stepDefinition, nodes, model, logsFile)
