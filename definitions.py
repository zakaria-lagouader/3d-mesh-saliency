import collections
import numpy as np

Geometry = collections.namedtuple("Geometry", "vertices, normals, faces, edges, adjacency")
Vertex = collections.namedtuple("Vertex",
                                "index,position,delta,normal,neighbouringFaceIndices,neighbouringVerticesIndices,rotationAxis,theta,color,edgeIndices")
Face = collections.namedtuple("Face",
                              "index,centroid,delta,vertices,verticesIndices,verticesIndicesSorted,faceNormal,area,edgeIndices,neighbouringFaceIndices,guidedNormal,rotationAxis,theta,color,neighbouringFaceIndices64")
Edge = collections.namedtuple("Edge", "index,vertices,verticesIndices,length,facesIndices,edgeNormal")

def computeRotation(vec, target):
    theta = (np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target)))
    axis_pre = np.cross(vec, target)
    if np.linalg.norm(axis_pre) != 0:
        axis = axis_pre / np.linalg.norm(axis_pre)

    if np.linalg.norm(axis_pre) == 0:
        axis_pre = np.cross(vec, np.roll(target, 1))
        axis = axis_pre / np.linalg.norm(axis_pre)

    return axis, theta

def rotatePatch(patch, axis, theta):
    if theta != 0:
        I = np.eye(3)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        # R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)
        R = I + np.sqrt(1 - (theta * theta)) * K + (1 - theta) * np.matmul(K, K)
        for fIndex in range(np.shape(patch)[0]):
            patch[fIndex,] = np.matmul(R, np.transpose(
                patch[fIndex,]))
    patch = np.transpose(patch)

    return patch

def cropList(listToProcess, maxElements):
    if len(listToProcess) < (maxElements):
        for i in range(0, maxElements - len(listToProcess)):
            listToProcess.append(listToProcess[0])
    if len(listToProcess) > (maxElements):
        listToProcess = listToProcess[:(maxElements)]
    return listToProcess

def neighboursByFace(Geom, faceIndex, numOfNeighbours):
    rings = [faceIndex]
    patchFaces = [faceIndex]
    crosschecklist = set()
    for i in patchFaces:
        for j in Geom.faces[i].neighbouringFaceIndices:
            if len(patchFaces) < numOfNeighbours:
                if j not in crosschecklist:
                    patchFaces.append(j)
                    crosschecklist.add(j)

    patchFaces = cropList(patchFaces, numOfNeighbours)
    return patchFaces, rings

def loadObj(filename):
    vertices = []
    normals = []
    faces = []
    edges = []
    adjacency = []
    with open(filename, newline='') as f:
        flines = f.readlines()
        # Read vertices
        indexCounter = 0;
        # print('Reading Vertices')
        for row in flines:
            if row[0] == 'v' and row[1] == ' ':
                line = row.rstrip()
                line = line[2:len(line)]
                coords = line.split()
                coords = list(map(float, coords))
                v = Vertex(
                    index=indexCounter,
                    position=np.asarray([coords[0], coords[1], coords[2]]),
                    delta=[],
                    normal=np.asarray([0.0, 0.0, 0.0]),
                    neighbouringFaceIndices=[],
                    neighbouringVerticesIndices=[],
                    theta=0.0,
                    edgeIndices=[],
                    rotationAxis=np.asarray([0.0, 0.0, 0.0]),
                    # color=np.asarray([coords[3], coords[4], coords[5]])
                    color=np.asarray([0.0, 0.0, 0.0])
                )
                indexCounter += 1;
                vertices.append(v)
        # Read Faces
        indexCounter = 0
        # print('Reading Faces')
        for row in flines:
            if row[0] == 'f':
                line = row.rstrip()
                line = line[2:len(line)]
                lineparts = line.strip().split()
                faceline = [];
                for fi in lineparts:
                    fi = fi.split('/')
                    faceline.append(int(fi[0]) - 1)
                f = Face(
                    index=indexCounter,
                    verticesIndices=[int(faceline[0]), int(faceline[1]), int(faceline[2])],
                    verticesIndicesSorted=[int(faceline[0]), int(faceline[1]), int(faceline[2])].sort(),
                    vertices=[],
                    centroid=np.asarray([0.0, 0.0, 0.0]),
                    delta=[],
                    faceNormal=np.asarray([0.0, 0.0, 0.0]),
                    edgeIndices=[],
                    area=0.0,
                    neighbouringFaceIndices=[],
                    guidedNormal=np.asarray([0.0, 0.0, 0.0]),
                    theta=0.0,
                    rotationAxis=np.asarray([0.0, 1.0, 0.0]),
                    color=np.asarray([0.0, 0.0, 0.0]),
                    neighbouringFaceIndices64=[]
                )
                indexCounter += 1;
                faces.append(f)

        # Which vertices are neighbouring to each vertex
        # print('Which vertices are neighbouring to each vertex')
        for idx_f, f in enumerate(faces):
            v0 = f.verticesIndices[0]
            v1 = f.verticesIndices[1]
            v2 = f.verticesIndices[2]
            if v1 not in vertices[v0].neighbouringVerticesIndices:
                vertices[v0].neighbouringVerticesIndices.append(v1)
            if v2 not in vertices[v0].neighbouringVerticesIndices:
                vertices[v0].neighbouringVerticesIndices.append(v2)
            if v0 not in vertices[v1].neighbouringVerticesIndices:
                vertices[v1].neighbouringVerticesIndices.append(v0)
            if v2 not in vertices[v1].neighbouringVerticesIndices:
                vertices[v1].neighbouringVerticesIndices.append(v2)
            if v0 not in vertices[v2].neighbouringVerticesIndices:
                vertices[v2].neighbouringVerticesIndices.append(v0)
            if v1 not in vertices[v2].neighbouringVerticesIndices:
                vertices[v2].neighbouringVerticesIndices.append(v1)

        # Which faces are neighbouring to each vertex
        # print('Which faces are neighbouring to each vertex')
        for idx_f, f in enumerate(faces):
            for idx_v, v in enumerate(f.verticesIndices):
                vertices[v].neighbouringFaceIndices.append(f.index)

        # print('Which faces are neighbouring to each face')
        for idx_v, v in enumerate(vertices):
            for idx_f in v.neighbouringFaceIndices:
                for jdx_f in v.neighbouringFaceIndices:
                    if idx_f != jdx_f:
                        common = set(faces[idx_f].verticesIndices) & set(faces[jdx_f].verticesIndices)
                        if len(common) == 2:
                            faces[idx_f].neighbouringFaceIndices.append(jdx_f)
                            faces[jdx_f].neighbouringFaceIndices.append(idx_f)

        for idx_f, fi in enumerate(faces):
            neighbouringFaceIndices = faces[idx_f].neighbouringFaceIndices
            faces[idx_f] = faces[idx_f]._replace(neighbouringFaceIndices=list(set(neighbouringFaceIndices)))

        ind_e = 0
        _edges = []
        for idx_v, vi in enumerate(vertices):
            neighbouringVerts = vertices[idx_v].neighbouringVerticesIndices
            for nv in neighbouringVerts:
                _edges.append([idx_v, nv])
        _edges = np.asarray(_edges)
        _edges.sort(axis=1)
        _edges = np.unique(_edges, axis=0)
        _edges = _edges.tolist()
        for idx_e, ei in enumerate(_edges):

            vertices[ei[0]].edgeIndices.append(idx_e)
            vertices[ei[1]].edgeIndices.append(idx_e)

            f1 = vertices[ei[0]].neighbouringFaceIndices
            f2 = vertices[ei[1]].neighbouringFaceIndices

            fs = list(set(f1).intersection(f2))
            E = Edge(
                index=ind_e,
                vertices=[],
                verticesIndices=[ei[0], ei[1]],
                length=0.0,
                facesIndices=fs,
                edgeNormal=np.asarray([1.0, 0.0, 0.0])
            )
            edges.append(E)
            for fsi in fs:
                faces[fsi].edgeIndices.append(idx_e)

        # print('Compute 64 neighbs')
        for idx_f, fi in enumerate(faces):
            patchFaces = [idx_f]
            crosschecklist = set()
            for i in patchFaces:
                for j in faces[i].neighbouringFaceIndices:
                    if len(patchFaces) < 64:
                        if j not in crosschecklist:
                            patchFaces.append(j)
                            crosschecklist.add(j)
            faces[idx_f] = faces[idx_f]._replace(neighbouringFaceIndices64=patchFaces)

    return Geometry(
        vertices=vertices,
        normals=normals,
        faces=faces,
        edges=edges,
        adjacency=[]
    )

def updateGeometryAttibutes(Geom, useGuided=False, numOfFacesForGuided=10, computeDeltas=False,
                            computeVertexNormals=True):
    for idx_f, f in enumerate(Geom.faces):
        v = [Geom.vertices[fvi] for fvi in Geom.faces[idx_f].verticesIndices]
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(vertices=v)
    for idx_f, f in enumerate(Geom.faces):
        vPos = [Geom.vertices[i].position for i in Geom.faces[idx_f].verticesIndices]
        vPos = np.asarray(vPos)
        centroid = np.mean(vPos, axis=0)
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(centroid=centroid)
    for idx_f, f in enumerate(Geom.faces):
        bc = Geom.faces[idx_f].vertices[1].position - Geom.faces[idx_f].vertices[2].position
        ba = Geom.faces[idx_f].vertices[0].position - Geom.faces[idx_f].vertices[2].position
        normal = np.cross(bc, ba)
        faceArea = 0.5 * np.linalg.norm(normal)
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(area=faceArea)
        if np.linalg.norm(normal) != 0:
            normalizedNormal = normal / np.linalg.norm(normal)
        else:
            normalizedNormal = np.asarray([0.0, 1.0, 0.0])
            print("Problem in " + str(idx_f))
        Geom.faces[idx_f].faceNormal[0] = normalizedNormal[0]
        Geom.faces[idx_f].faceNormal[1] = normalizedNormal[1]
        Geom.faces[idx_f].faceNormal[2] = normalizedNormal[2]
        if np.linalg.norm(normalizedNormal) == 0:
            print('Warning')

    if computeVertexNormals:
        for idx_v, v in enumerate(Geom.vertices):
            normal = np.asarray([0.0, 0.0, 0.0])
            for idx_f, f in enumerate(Geom.vertices[idx_v].neighbouringFaceIndices):
                normal += Geom.faces[f].faceNormal
            normal = normal / len(Geom.vertices[idx_v].neighbouringFaceIndices)
            normal = normal / np.linalg.norm(normal)
            Geom.vertices[idx_v].normal[0] = normal[0]
            Geom.vertices[idx_v].normal[1] = normal[1]
            Geom.vertices[idx_v].normal[2] = normal[2]

    if useGuided:
        numOfFaces_ = numOfFacesForGuided
        patches = []
        for i in range(0, len(Geom.faces)):
            p = neighboursByFace(Geom, i, numOfFaces_)
            patches.append(p)
        for idx_f, f in enumerate(Geom.faces):
            if idx_f != f.index:
                print('Maybe prob', f.index)
            selectedPatches = []
            for p in patches:
                if f.index in p:
                    selectedPatches.append(p)
            patchFactors = []
            for p in selectedPatches:
                patchFaces = [Geom.faces[i] for i in p]
                patchNormals = [pF.faceNormal for pF in patchFaces]
                normalsDiffWithinPatch = [np.linalg.norm(patchNormals[0] - p, 2) for p in patchNormals]
                maxDiff = max(normalsDiffWithinPatch)
                patchNormals = np.asarray(patchNormals)
                M = np.matmul(np.transpose(patchNormals), patchNormals)
                w, v = np.linalg.eig(M)
                eignorm = np.linalg.norm(np.diag(v))
                patchFactor = eignorm * maxDiff
                patchFactors.append(patchFactor)
            minIndex = np.argmin(np.asarray(patchFactors))
            p = selectedPatches[minIndex]
            patchFaces = [Geom.faces[i] for i in p]
            weightedNormalFactors = [pF.area * pF.faceNormal for pF in patchFaces]
            weightedNormalFactors = np.asarray(weightedNormalFactors)
            weightedNormal = np.mean(weightedNormalFactors, axis=0)
            weightedNormal = weightedNormal / np.linalg.norm(weightedNormal)
            Geom.faces[f.index] = Geom.faces[f.index]._replace(guidedNormal=weightedNormal)

    if computeDeltas:
        for idx_v, v in enumerate(Geom.vertices):
            neibs = Geom.vertices[idx_v].neighbouringVerticesIndices
            vPos = np.asarray([Geom.vertices[i].position for i in neibs])
            sSum = np.sum(vPos, axis=0) / len(neibs)
            computedDelta = Geom.vertices[idx_v].position - sSum
            Geom.vertices[idx_v] = Geom.vertices[idx_v]._replace(delta=computedDelta)
