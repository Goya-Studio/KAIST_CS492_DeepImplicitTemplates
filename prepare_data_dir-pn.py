import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' 
import trimesh
from tqdm import tqdm
import numpy as np
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere, BadMeshException 
from util import ensure_directory
from multiprocessing import Pool
from rendering.math import get_rotation_matrix

DIRECTORY_MODEL = 'ShapeNetCore.v2/04256520' #sofa
MODEL_EXTENSION = '.obj'

VOXEL_RESOLUTION = 128  # 8 16 32 64

DIRECTORY_SDF = 'data/sdf/'

CREATE_VOXELS = None


CREATE_SDF_CLOUDS = True   #True
SDF_CLOUD_SAMPLE_SIZE = 200000

ROTATION = None # get_rotation_matrix(90, axis='x')

#확장자에 맞는 파일을 가져온다.
def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)
#b)
def get_npy_filename(model_filename, qualifier=''):
    return DIRECTORY_SDF + base_dir + qualifier + '.npz'
    #return DIRECTORY_SDF + model_filename[len(DIRECTORY_MODELS):-len(MODEL_EXTENSION)] + qualifier + '.npy'


def get_voxel_filename(model_filename):
    return get_npy_filename(model_filename, '')

#a)
def get_sdf_cloud_filename(model_filename):
    return get_npy_filename(base_dir) # 수정사항

def get_bad_mesh_filename(model_filename):
    return DIRECTORY_SDF + model_filename[len(DIRECTORY_MODELS):-len(MODEL_EXTENSION)] + i + '.badmesh' #수정사항

def mark_bad_mesh(model_filename):
    filename = get_bad_mesh_filename(model_filename)
    ensure_directory(os.path.dirname(filename))            
    open(filename, 'w').close()

def is_bad_mesh(model_filename):
    return os.path.exists(get_bad_mesh_filename(model_filename))
#2)
def process_model_file(filename):
    voxels_filename = get_voxel_filename(filename)
#    sdf_cloud_filename = get_sdf_cloud_filename(filename)
    sdf_cloud_filename = get_sdf_cloud_filename(filename)
    #check = 0
    print("process_model_file - start")

    if is_bad_mesh(filename):
        print("process_model_file - bad??")
        # check = 1
        return
    if not (CREATE_VOXELS and not os.path.isfile(voxels_filename) or CREATE_SDF_CLOUDS and not os.path.isfile(sdf_cloud_filename)):
        print("process_model_file - not?")
        return
    print("process_model_file - end", filename)
    #mesh = trimesh.load(filename)
    mesh = trimesh.load(filename)
    print( "first mesh",mesh)
    if ROTATION is not None:
        mesh.apply_transform(ROTATION)
    #mesh = scale_to_unit_sphere(mesh)
    print(filename)
    surface_point_cloud = get_surface_point_cloud(mesh)   #### 여기가 문제임,,, !
    # 이곳을 지나야 SDF 파일이 만들어진다!
    if CREATE_SDF_CLOUDS:
        print("///CREATE_SDF_CLOUDS///")
        try:
            print("process_model_file - createsdf1") #여기서 error 발생하는 거 같음
            #mesh, surface_point_method='scan',
            # scan_count=100, scan_resolution=400, sample_point_count=10000000,
            points, sdf = surface_point_cloud.sample_sdf_near_surface(number_of_points=SDF_CLOUD_SAMPLE_SIZE) #, sign_method='depth', min_size=0.015)
            sdf = sdf.reshape(-1,1)
            #points, sdf = sample_sdf_near_surface(mesh, number_of_points=SDF_CLOUD_SAMPLE_SIZE, sign_method='depth', min_size=0.015)
            print("process_model_file - createsdf2")
            combined = np.concatenate((points, sdf), axis=1)
            print("process_model_file - createsdf3")
            ensure_directory(os.path.dirname(sdf_cloud_filename))
            print("process_model_file - createsdf4")
            print("SDF")
#             print(sdf)
#             print(np.shape(sdf))
#             print(np.shape(points))
#             print(np.shape(combined))
            combined_pos = []
            combined_neg = []
            for i in range(sdf.size):
                combined_i = combined[i,:]
                if sdf[i,0] >=0:
                    if combined_pos ==[]:
                        combined_pos = combined_i
                    else:
                        combined_pos = np.append(combined_pos,combined_i, axis=0)
                elif sdf[i,0] <0:
                    if combined_neg ==[]:
                        combined_neg = combined_i
                    else:
                        combined_neg = np.append(combined_neg,combined_i, axis=0)
            print(np.shape(combined_neg))
            combined_neg = combined_neg.reshape(-1,4)
            combined_pos = combined_pos.reshape(-1,4)
            np.savez(sdf_cloud_filename, pos = combined_pos, neg= combined_neg) #points, sdf[:, np.newaxis])
            print("process_model_file - createsdf5")
            
        except BadMeshException:
            tqdm.write("Skipping bad mesh. ({:s})".format(filename))
            mark_bad_mesh(filename)
            return

    if CREATE_VOXELS:
        # print("CREATE_VOXELS")
        try:
            voxels = surface_point_cloud.get_voxels(voxel_resolution=VOXEL_RESOLUTION, use_depth_buffer=True)
            ensure_directory(os.path.dirname(voxels_filename))
            np.save(voxels_filename, voxels)
        except BadMeshException:
            tqdm.write("Skipping bad mesh. ({:s})".format(filename))
            mark_bad_mesh(filename)
            return

#1)
def process_model_files():
    ensure_directory(DIRECTORY_SDF)
    files = list(get_model_files())

    # print(files)
    
    worker_count = os.cpu_count() // 2
    print("Using {:d} processes.".format(worker_count))  ## cpu로 동시에 돌아감,,
    pool = Pool(worker_count)

    print("hi1")
    print(base_dir)
    progress = tqdm(total=len(files))
    def on_complete(*_):
        progress.update()
    print(files)
    for filename in files:
        #2)
#        print(base_dir)
        pool.apply_async(process_model_file, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()
#3)
def combine_pointcloud_files():
    import torch
    print("Combining SDF point clouds...")
    npy_files = sorted([get_sdf_cloud_filename(f) for f in get_model_files()])


    N = len(npy_files)
    
    print("npy_files", npy_files)
    points = torch.zeros((N * SDF_CLOUD_SAMPLE_SIZE, 3))
    sdf = torch.zeros((N * SDF_CLOUD_SAMPLE_SIZE,1))

    print("Saving combined SDF clouds...")

def list_dir(pth):
    fol_lists = os.listdir(pth)
    len_fol = len(fol_lists)
    folder = []
    for fol_list in fol_lists:
        fol_path = os.path.join(pth, fol_list)
        if os.path.isdir(fol_path):
            folder.append(fol_path)
    return folder

if __name__ == '__main__':
    dirs = list_dir(DIRECTORY_MODEL)
    i1 = 0
    
    for DIRECTORY_MODELS in dirs:
        print("=======")
        i1 = i1+1
        i = str(i1)
        base_dir = os.path.basename(os.path.normpath(DIRECTORY_MODELS))
        process_model_files()
        if CREATE_SDF_CLOUDS:
            combine_pointcloud_files()