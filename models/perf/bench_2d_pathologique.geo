SetFactory("OpenCASCADE");
lc = 0.035;

Rectangle(1) = {0, 0, 0, 3.2, 2.2};
Disk(2) = {0.75, 0.65, 0, 0.22, 0.22};
Disk(3) = {2.35, 1.45, 0, 0.28, 0.28};
Disk(4) = {1.55, 1.10, 0, 0.18, 0.18};
Rectangle(5) = {1.30, 0.00, 0, 0.10, 0.82};
Rectangle(6) = {1.30, 1.20, 0, 0.10, 1.00};
Rectangle(7) = {0.30, 1.02, 0, 0.86, 0.09};
Rectangle(8) = {1.85, 0.45, 0, 0.75, 0.08};
Rectangle(9) = {2.72, 0.22, 0, 0.16, 0.22};
Rectangle(10) = {0.36, 1.55, 0, 0.20, 0.18};
Rectangle(11) = {2.05, 1.78, 0, 0.22, 0.14};
Disk(12) = {1.02, 0.24, 0, 0.09, 0.09};
Disk(13) = {2.82, 1.02, 0, 0.11, 0.11};

air[] = BooleanDifference{ Surface{1}; Delete; }{ Surface{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}; };

Physical Surface(5) = {air[]};
Physical Surface(1) = {5, 6, 7, 8};
Physical Surface(2) = {2, 3};
Physical Surface(6) = {9, 11};
Physical Surface(7) = {10};
Physical Surface(8) = {12, 13};

Mesh.MeshSizeMin = lc * 0.45;
Mesh.MeshSizeMax = lc;
Mesh.Algorithm = 6;
