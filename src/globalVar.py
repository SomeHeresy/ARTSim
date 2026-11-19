# Author: Adam Corbier (@Ad2Am2)

GMatrix = []
CMatrix = []
IVector = []
IVectorVector = []
nodes = []
model = []
modelPrepared = False
baseTemp = 318.5

# New vars
# heat capacity of each node (1-D, replaces the N x N CMatrix)
CVector = []
# sparse G: row of each entry
G_rows = []
# sparse G: column of each entry
G_cols = []
# sparse G: value of each entry
G_vals = []
# (layer, chiplet, unit) -> center node index
centerLookup = {}
# (layer, chiplet, unit) -> side ground node indexes
groundLookup = {}
# (layer, chiplet, unit) -> top/bottom ground node indexes
ground3DLookup = {}
