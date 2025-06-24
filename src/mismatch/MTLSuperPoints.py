import argparse
import os
import numpy as np
import pandas as pd
import vtk
from collections import defaultdict
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pymetis
from glob import glob

# Seed for reproducibility
np.random.seed(42)

def read_vtk_mesh(file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def vtk_get_triangles(polydata):
    return vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

def get_cell_data(polydata, array_name):
    array = polydata.GetCellData().GetArray(array_name)
    return vtk_to_numpy(array) if array is not None else None

def get_point_data(polydata, array_name):
    array = polydata.GetPointData().GetArray(array_name)
    return vtk_to_numpy(array) if array is not None else None

def write_vtk_mesh_with_partitions(vertices, triangles, membership, labels, filename):
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(vertex)

    triangles_vtk = vtk.vtkCellArray()
    for triangle in triangles:
        triangle_vtk = vtk.vtkTriangle()
        triangle_vtk.GetPointIds().SetId(0, triangle[0])
        triangle_vtk.GetPointIds().SetId(1, triangle[1])
        triangle_vtk.GetPointIds().SetId(2, triangle[2])
        triangles_vtk.InsertNextCell(triangle_vtk)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles_vtk)

    partition_array = vtk.vtkIntArray()
    partition_array.SetName("Partition")
    partition_array.SetNumberOfComponents(1)
    partition_array.SetNumberOfTuples(len(triangles))
    for i, partition in enumerate(membership):
        partition_array.SetValue(i, int(partition))

    polydata.GetCellData().SetScalars(partition_array)

    label_array = numpy_to_vtk(labels, deep=True, array_type=vtk.VTK_INT)
    label_array.SetName("Label")
    polydata.GetCellData().AddArray(label_array)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

    print(f"\u2705 VTK saved: {filename}")

def compute_mean_thickness(subject_thickness_path, triangles, membership, ID, MRIDATE, output_txt_path, num_partitions):
    polydata = read_vtk_mesh(subject_thickness_path)
    voronoi = get_point_data(polydata, "VoronoiRadius")
    if voronoi is None:
        raise ValueError(f"VoronoiRadius not found in: {subject_thickness_path}")

    partition_values = [[] for _ in range(num_partitions)]
    for i, tri in enumerate(triangles):
        if membership[i] < 0:
            continue
        values = voronoi[tri]
        partition_values[membership[i]].extend(values[~np.isnan(values)] * 2)

    results = []
    for part in partition_values:
        results.append(np.mean(part) if len(part) > 0 else np.nan)

    with open(output_txt_path, 'w') as f:
        f.write(f"{ID} {MRIDATE} " + " ".join(f"{v:.4f}" if not np.isnan(v) else "NA" for v in results) + "\n")

    print(f"\u2705 Mean thickness saved: {output_txt_path}")

def load_partition_txts(folder, hemi):
    files = sorted(glob(os.path.join(folder, f"mean_thickness_*_{hemi}.txt")))
    if not files:
        raise FileNotFoundError(f"No mean_thickness files found for hemisphere '{hemi}' in: {folder}")
    dfs = []
    for file in files:
        df = pd.read_csv(file, sep=" ", header=None)
        df.columns = ["ID", "MRIDATE"] + [f"x_{hemi}_{i}" for i in range(df.shape[1] - 2)]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def merge_thickness(input_folder, output_csv):
    left_df = load_partition_txts(input_folder, "left")
    right_df = load_partition_txts(input_folder, "right")
    merged = pd.merge(left_df, right_df, on=["ID", "MRIDATE"], how="inner")
    final_df = merged
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Final merged CSV saved: {output_csv}")
    return merged

def run_partition_pipeline(subject_csv, template_path, output_dir, num_partitions, hemi):
    class Args:
        pass
    args = Args()
    args.subject_csv = subject_csv
    args.template_path = template_path
    args.output_dir = output_dir
    args.num_partitions = num_partitions
    args.hemi = hemi
    main_partition(args)

def main_partition(args):
    df = pd.read_csv(args.subject_csv)
    template_polydata = read_vtk_mesh(args.template_path)
    triangles = vtk_get_triangles(template_polydata)
    labels = get_cell_data(template_polydata, 'label')
    if labels is None:
        raise ValueError("Label array not found in the template VTK file")

    valid_triangle_indices = np.where(labels != 0)[0]
    filtered_triangles = triangles[valid_triangle_indices]
    filtered_labels = labels[valid_triangle_indices]

    unique_labels = np.unique(filtered_labels)
    total_valid_triangles = len(filtered_triangles)

    total_partitions = args.num_partitions
    label_partitions = {}
    for label in unique_labels:
        count = np.sum(filtered_labels == label)
        label_partitions[label] = max(1, round((count / total_valid_triangles) * total_partitions))

    diff = total_partitions - sum(label_partitions.values())
    sorted_labels = sorted(label_partitions, key=label_partitions.get, reverse=True)
    for i in range(abs(diff)):
        if diff > 0:
            label_partitions[sorted_labels[i % len(sorted_labels)]] += 1
        elif diff < 0 and label_partitions[sorted_labels[i % len(sorted_labels)]] > 1:
            label_partitions[sorted_labels[i % len(sorted_labels)]] -= 1

    partition_offset = 0
    final_membership = np.full(len(triangles), -1)

    summary_file = os.path.join(args.output_dir, f"partition_summary_{args.hemi}.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write("Partition Summary for Each Label:\n\n")

    for label in unique_labels:
        label_triangle_indices = np.where(labels == label)[0]
        label_triangle_indices = np.sort(label_triangle_indices)
        label_triangles = triangles[label_triangle_indices]

        vertex_to_triangles = defaultdict(list)
        adj = []
        for tri_idx, triangle in enumerate(label_triangles):
            for vertex in triangle:
                vertex_to_triangles[vertex].append(tri_idx)

        for v in range(len(label_triangles)):
            tri_set = set(label_triangles[v])
            neighbors = set()
            for vertex in tri_set:
                for nb in vertex_to_triangles[vertex]:
                    if nb != v and len(tri_set.intersection(label_triangles[nb])) >= 2:
                        neighbors.add(nb)
            adj.append(list(neighbors))

        n_parts = label_partitions[label]
        _, membership = pymetis.part_graph(n_parts, adjacency=adj)
        global_membership = np.array(membership) + partition_offset
        final_membership[label_triangle_indices] = global_membership

        with open(summary_file, 'a') as f:
            f.write(f"Label {int(label)} -> Partitions: {sorted(set(global_membership))}\n")

        partition_offset += n_parts

    vertices = vtk_to_numpy(template_polydata.GetPoints().GetData())
    for _, row in df.iterrows():
        ID, MRIDATE = row['ID'], row['MRIDATE']
        out_file = os.path.join(args.output_dir, f"{ID}_{MRIDATE}_partitioned_{args.hemi}.vtk")
        write_vtk_mesh_with_partitions(vertices, triangles, final_membership, labels, out_file)

        thickness_path = row['Path']
        if not os.path.exists(thickness_path):
            print(f"Warning: missing thickness file for {ID} {MRIDATE}")
            continue

        out_txt = os.path.join(args.output_dir, f"mean_thickness_{ID}_{MRIDATE}_{args.hemi}.txt")
        compute_mean_thickness(thickness_path, triangles, final_membership, ID, MRIDATE, out_txt, total_partitions)

def create_MTLSuperPoints(left_csv, right_csv, template_left, template_right, output_dir, num_partitions, final_csv):
    run_partition_pipeline(left_csv, template_left, output_dir, num_partitions, "left")
    run_partition_pipeline(right_csv, template_right, output_dir, num_partitions, "right")
    #merge_thickness(output_dir, final_csv)
    final_df = merge_thickness(output_dir, final_csv)
    return {"final_df": final_df}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_csv", required=True)
    parser.add_argument("--right_csv", required=True)
    parser.add_argument("--template_left", required=True)
    parser.add_argument("--template_right", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_partitions", type=int, default=50)
    parser.add_argument("--final_csv", required=True)
    args = parser.parse_args()

    create_MTLSuperPoints(
        args.left_csv,
        args.right_csv,
        args.template_left,
        args.template_right,
        args.output_dir,
        args.num_partitions,
        args.final_csv
    )

if __name__ == "__main__":
    main()
