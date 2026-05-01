SetFactory("OpenCASCADE");
lc = 1.25;
L = 5.0;
W = 4.0;
H = 2.4;
e = 0.22;

Box(1) = {0, 0, 0, L, W, e};
Box(2) = {0, 0, e, L, e, H-e};
Box(3) = {0, W-e, e, L, e, H-e};
Box(4) = {0, e, e, e, W-2*e, H-e};
Box(5) = {L-e, e, e, e, W-2*e, H-e};
Box(6) = {2.2, e, e, e, 2.2, H-e};
Box(7) = {0.8, 0.9, e, 0.55, 0.55, 0.55};
Box(8) = {3.2, 2.4, e, 0.35, 0.35, 0.35};
Box(9) = {1.6, 2.6, e, 0.45, 0.18, 0.18};

Physical Volume(1) = {1};
Physical Volume(2) = {2, 3, 4, 5, 6};
Physical Volume(6) = {7};
Physical Volume(7) = {8};
Physical Volume(8) = {9};

Mesh.MeshSizeMin = lc * 0.6;
Mesh.MeshSizeMax = lc;
