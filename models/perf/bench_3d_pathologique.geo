SetFactory("OpenCASCADE");
lc = 0.38;

Box(1) = {0, 0, 0, 4.8, 3.6, 2.8};
Sphere(2) = {0.95, 0.90, 1.05, 0.32};
Sphere(3) = {2.45, 1.80, 1.42, 0.42};
Sphere(4) = {3.75, 2.55, 1.70, 0.28};
Box(5) = {0.35, 0.25, 0.15, 0.28, 3.10, 2.15};
Box(6) = {1.62, 0.20, 0.15, 0.24, 1.25, 2.05};
Box(7) = {1.62, 2.10, 0.15, 0.24, 1.20, 2.05};
Box(8) = {3.75, 0.55, 0.25, 0.42, 0.42, 0.65};
Box(9) = {2.78, 2.16, 0.25, 0.35, 0.35, 0.55};
Box(10) = {1.05, 2.70, 0.25, 0.52, 0.18, 0.18};

air[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2, 3, 4, 5, 6, 7, 8, 9, 10}; };

Physical Volume(5) = {air[]};
Physical Volume(1) = {5, 6, 7};
Physical Volume(2) = {2, 3, 4};
Physical Volume(6) = {8};
Physical Volume(7) = {9};
Physical Volume(8) = {10};

Mesh.MeshSizeMin = lc * 0.45;
Mesh.MeshSizeMax = lc;
Mesh.Algorithm3D = 1;
