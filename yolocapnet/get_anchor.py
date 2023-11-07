
import os
import argparse

import numpy as np
import sys
import math
import random

from data_loader import DataLoader

def intersect_over_union(x, centroids):
    similarities = []
    num = len(centroids)

    for centroid in centroids:
        cen_width, cen_height = centroid
        width, height = x

        if cen_width >= width and cen_height >= height:
            similarity = width * height / (cen_width * cen_height)
        elif cen_width>=width and cen_height<=height:
            similarity = cen_width * cen_height / (width * height)
        elif cen_width<=width and cen_height>=height:
            similarity = cen_width * cen_height / (width * height)
        else:
            similarity = (cen_width * cen_height) / (width * height)

        similarities.append(similarity)
    
    return np.array(similarities)

def avg_iou(x, centroids):
    n, d = x.shape
    sum = 0.

    for i in range(x.shape[0]):
        sum += max(intersect_over_union(x[i], centroids))

    return sum / n

def write_to_file(centeroids, x, file):
    f = open(file, 'w')
    anchors = centeroids.copy()
    # print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= 416/32
        anchors[i][1] *= 416/32
    
    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%.2f,%.2f, '%(anchors[i][0], anchors[i][1]))
    
    f.write('%.2f,%.2f\n'%(anchors[sorted_indices[-1]][0], anchors[sorted_indices[-1]][1]))

    f.write('%f\n'%(avg_iou(x, centeroids)))
    print()

def k_means(x, centeroids, epsilon, out_file):
    n = x.shape[0]
    num, dim = centeroids.shape
    iterations = 0
    prev_ass = np.ones(n) * (-1)
    iter = 0
    old = np.zeros((n, num))

    while True:
        D = []
        iter += 1
        for i in range(n):
            D.append(1 - intersect_over_union(x[i], centeroids))
        
        D = np.array(D)

        print(f"Iteration #{iter}: distances = {np.sum(np.abs(old - D))}")

        assignments = np.argmin(D, axis=1)

        if (assignments == prev_ass).all():
            print(f"Centeroids: {centeroids}")
            write_to_file(centeroids, x, out_file)
            return

        centroid_sum = np.zeros((num, dim), np.float32)
        for i in range(n):
            centroid_sum[assignments[i]] += x[i]
        
        for m in range(num):
            centeroids[m] = centroid_sum[m]/(np.sum(assignments==m))
        
        prev_ass = assignments.copy()
        old = D.copy()

def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Path to file list (e.g. \\path\\to\\file\\train_collection.txt)\n')
    parser.add_argument('-output_dir', help="Path to desired output directory\n")
    parser.add_argument('-n_clusters', default=0, type=int, help="Number of clusters\n")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exists_ok=True)

    loader = DataLoader()
    train_data = loader.load_train_file(train_file=args.train)
    dimensions = []

    for sample in train_data:
        w,h = sample[4:]
        dimensions.append(tuple(map(float, (w,h))))
    
    dimensions = np.array(dimensions)

    epsilon = 0.005

    if args.n_cluster == 0:
        for n_clusters in range(1,11):
            anchor_file = os.path.join(args.output_dir, 'anchors%d.txt'%(n_clusters))
            indices = [random.randrange(dimensions.shape[0]) for i in range(n_clusters)]
            centeroids = dimensions[indices]
            k_means(dimensions, centeroids, epsilon, anchor_file)
            print('centeroids shape: ', centeroids.shape)
    else:
        anchor_file = os.path.join(args.output_dir, 'anchors%d.txt'%(args.n_clusters))
        indices = [random.randrange(dimensions.shape[0]) for i in range(args.n_clusters)]
        centeroids = dimensions[indices]
        k_means(dimensions, centeroids, epsilon, anchor_file)
        print('centeroids shape: ', centeroids.shape)
        

if __name__ == '__main__':
    main(sys.argv)