import trimesh
import argparse
import numpy as np
from definitions import *
from keras.models import load_model
import hilbertcurve.hilbertcurve.hilbertcurve as hb

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--patchSize", type=int, help="Patch size")
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--mesh_path", type=str, help="Mesh name")
args = parser.parse_args()

# Configuration
patchSize = args.patchSize if args.patchSize else 16
numOfElements = patchSize * patchSize
target = np.asarray([0.0, 1.0, 0.0]) #Rotation target
model_name = args.model_name if args.model_name else "new-model-16.h5"
mesh_path = args.mesh_path if args.mesh_path else "3d-meshes/vase_decimated.obj"

# Hibert curve
p = patchSize
N = 2
hilbert_curve = hb.HilbertCurve(p, N)
I2HC=np.empty((p*p,2))
HC2I=np.empty((p,p))
hCoords=[]
for ii in range(p*p):
    h=hilbert_curve.coordinates_from_distance(ii)
    hCoords.append(h)
    I2HC[ii,:]=h
    HC2I[h[0],h[1]]=ii
I2HC=I2HC.astype(int)
HC2I=HC2I.astype(int)

# Load the model
model = load_model(f"saved-models/{model_name}")

def process_mesh(file_name):
    print(f"Processing {file_name}")
    mModel = loadObj(file_name)
    
    updateGeometryAttibutes(
        mModel, 
        numOfFacesForGuided=numOfElements, 
        computeVertexNormals=False
    )

    num_faces = len(mModel.faces)
    patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(num_faces)]
    train_data = np.empty((num_faces, patchSize, patchSize, 3), dtype=np.float32)

    for idx, patch in enumerate(patches):
        if idx % 2000 == 0:
            print(f"{file_name}: {idx / 200}%")
        
        patch_faces = [mModel.faces[j] for j in patch]
        normals = np.array([face.faceNormal for face in patch_faces])
        
        # Calculate rotation vector
        vec = np.mean([face.area * face.faceNormal for face in patch_faces], axis=0)
        vec /= np.linalg.norm(vec)
        axis, theta = computeRotation(vec, target)
        normals = rotatePatch(normals, axis, theta)
        
        # Reshape normals and apply transformations
        normals_reshaped = np.zeros((patchSize, patchSize, 3), dtype=np.float32)
        for hci in range(I2HC.shape[0]):
            i, j = I2HC[hci]
            normals_reshaped[i, j, :] = normals[:, HC2I[i, j]]
        
        # Normalize the data
        train_data[idx] = (normals_reshaped + 1) / 2

    return model.predict(train_data)


if __name__ == "__main__":
    mesh = trimesh.load(mesh_path)
    saliency_map = process_mesh(mesh_path)
    mesh.visual.face_colors = trimesh.visual.interpolate(saliency_map, color_map='jet')
    mesh.show()